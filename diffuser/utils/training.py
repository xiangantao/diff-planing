import copy
import os

import einops
import torch
import torch.nn.functional as F
from ml_logger import logger

import diffuser

from .arrays import apply_dict, batch_to_device, to_device, to_np
from .timer import Timer


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        eval_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device="cuda",
        save_checkpoints=False,
        # ---------------- Q_pattern (obs-diffusion) ----------------
        transition_q_lr: float = 1e-4,
        pattern_q_lr: float = 1e-4,
        rollout_every: int = 0,
        rollout_ddim_steps: int = 15,
        rollout_batch_size: int = 4,
        rollout_use_cfg: bool = True,
        rollout_policy_weight: float = 0.1,
        rollout_bc_weight: float = 1.0,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema

        assert (
            eval_freq % save_freq == 0
        ), f"eval_freq must be a multiple of save_freq, but got {eval_freq} and {save_freq} respectively"
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.eval_freq = eval_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        if dataset is not None:
            self.dataloader = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=train_batch_size,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )
            self.dataloader_vis = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )

        self.renderer = renderer
        # ---------------- optimizers ----------------
        self.use_q_pattern = bool(getattr(diffusion_model, "use_q_pattern", False))
        self.rollout_every = int(rollout_every)
        self.rollout_ddim_steps = int(rollout_ddim_steps)
        self.rollout_batch_size = int(rollout_batch_size)
        self.rollout_use_cfg = bool(rollout_use_cfg)
        self.rollout_policy_weight = float(rollout_policy_weight)
        self.rollout_bc_weight = float(rollout_bc_weight)

        self.transition_q_optimizer = None
        self.pattern_q_optimizer = None

        if self.use_q_pattern:
            assert hasattr(diffusion_model, "transition_q")
            assert hasattr(diffusion_model, "pattern_encoder")
            assert hasattr(diffusion_model, "pattern_q")

            # Exclude critic params from the policy optimizer (same philosophy as cc_trainer)
            critic_param_ids = set()
            for p in diffusion_model.transition_q.parameters():
                critic_param_ids.add(id(p))
            for p in diffusion_model.pattern_q.parameters():
                critic_param_ids.add(id(p))

            policy_params = [p for p in diffusion_model.parameters() if id(p) not in critic_param_ids]
            self.optimizer = torch.optim.Adam(policy_params, lr=train_lr)
            self.transition_q_optimizer = torch.optim.Adam(
                diffusion_model.transition_q.parameters(), lr=transition_q_lr
            )
            self.pattern_q_optimizer = torch.optim.Adam(
                diffusion_model.pattern_q.parameters(), lr=pattern_q_lr
            )
        else:
            self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.evaluator = None
        self.device = train_device

    def _slice_batch(self, batch, n: int):
        out = {}
        for k, v in batch.items():
            if k == "cond":
                out["cond"] = {ck: cv[:n] for ck, cv in v.items()}
            elif torch.is_tensor(v):
                out[k] = v[:n]
            else:
                out[k] = v
        return out

    def _transition_q_loss(self, batch):
        """
        Supervised transition critic training (fast first version):
            Q_tau(h_t, s_{t+1}) ~= returns
        """
        if (not self.use_q_pattern) or ("returns" not in batch):
            return None

        action_dim = self.model.action_dim
        H = self.model.history_horizon

        obs = batch["x"][..., action_dim:]  # [B, T, A, D_obs]
        h_obs = obs[:, : H + 1]  # [B, H+1, A, D_obs]
        s_tp1 = obs[:, H + 1]  # [B, A, D_obs]

        # returns: [B, 1, A] -> scalar label per batch item
        target = batch["returns"].mean(dim=-1)  # [B,1]
        pred = self.model.transition_q(h_obs, s_tp1)
        return F.mse_loss(pred, target)

    def _qpattern_rollout_losses(self, batch):
        """
        DDIM rollout (differentiable) + policy loss via PatternQ + rollout BC via inv_model,
        plus a distillation loss for pattern_q (trained separately).
        """
        assert self.use_q_pattern
        bsz = min(self.rollout_batch_size, batch["x"].shape[0])
        b = self._slice_batch(batch, bsz)

        action_dim = self.model.action_dim
        H = self.model.history_horizon

        # rollout obs with diffusion chain
        obs_plan, chain_obs = self.model.conditional_sample_grad(
            cond=b["cond"],
            returns=b.get("returns", None),
            env_ts=b.get("env_ts", None),
            attention_masks=b.get("attention_masks", None),
            n_ddim_steps=self.rollout_ddim_steps,
            use_cfg=self.rollout_use_cfg,
            return_diffusion=True,
        )

        # history window for Q inputs
        h_obs = obs_plan[:, : H + 1]
        s_tp1_plan = obs_plan[:, H + 1]

        # pattern latent from the *full* obs diffusion chain
        pattern_latent = self.model.pattern_encoder(chain_obs)

        # ---- policy loss: maximize q_pattern, but do NOT update pattern_q params here
        pattern_q_params = list(self.model.pattern_q.parameters())
        prev_req = [p.requires_grad for p in pattern_q_params]
        for p in pattern_q_params:
            p.requires_grad_(False)

        q_pattern = self.model.pattern_q(h_obs, pattern_latent)
        policy_loss = -q_pattern.mean()

        for p, r in zip(pattern_q_params, prev_req):
            p.requires_grad_(r)

        # ---- rollout BC via inv-dynamics (use dataset actions as targets)
        actions_gt = b["x"][..., :action_dim]
        x_roll = torch.cat([actions_gt, obs_plan], dim=-1)
        inv_loss_roll, _ = self.model.compute_inv_loss(
            x_roll,
            b["loss_masks"],
            legal_actions=None,
        )

        # ---- distillation for pattern_q (teacher: transition_q on rollout next state)
        with torch.no_grad():
            q_label = self.model.transition_q(h_obs.detach(), s_tp1_plan.detach()).detach()
        distill_pred = self.model.pattern_q(h_obs.detach(), pattern_latent.detach())
        pattern_q_loss = F.mse_loss(distill_pred, q_label)

        info = {
            "q_pattern": q_pattern.detach().mean(),
            "rollout_inv_loss": inv_loss_roll.detach(),
            "pattern_q_loss": pattern_q_loss.detach(),
        }

        return policy_loss, inv_loss_roll, pattern_q_loss, info

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def finish_training(self):
        if self.step % self.save_freq == 0:
            self.save()
        if self.eval_freq > 0 and self.step % self.eval_freq == 0:
            self.evaluate()
        if self.evaluator is not None:
            del self.evaluator

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        timer = Timer()
        for _ in range(n_train_steps):
            pattern_q_loss_to_step = None
            rollout_info = {}

            # zero grads for each optimizer
            self.optimizer.zero_grad(set_to_none=True)
            if self.transition_q_optimizer is not None:
                self.transition_q_optimizer.zero_grad(set_to_none=True)

            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(**batch)

                # transition_q supervised update (cheap, every step)
                if self.transition_q_optimizer is not None:
                    q_loss = self._transition_q_loss(batch)
                    if q_loss is not None:
                        (q_loss / self.gradient_accumulate_every).backward()
                        infos["transition_q_loss"] = q_loss.detach()

                # low-frequency rollout loss (DDIM-15) for Q_pattern + rollout BC
                if (
                    self.use_q_pattern
                    and self.rollout_every > 0
                    and (self.step % self.rollout_every == 0)
                    and i == 0
                ):
                    policy_loss, rollout_inv_loss, pattern_q_loss, roll_infos = (
                        self._qpattern_rollout_losses(batch)
                    )
                    rollout_info = roll_infos
                    loss = (
                        loss
                        + self.rollout_policy_weight * policy_loss
                        + self.rollout_bc_weight * rollout_inv_loss
                    )
                    pattern_q_loss_to_step = pattern_q_loss

                (loss / self.gradient_accumulate_every).backward()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.transition_q_optimizer is not None:
                self.transition_q_optimizer.step()
                self.transition_q_optimizer.zero_grad(set_to_none=True)

            # train pattern_q only when we ran a rollout
            if self.pattern_q_optimizer is not None and pattern_q_loss_to_step is not None:
                self.pattern_q_optimizer.zero_grad(set_to_none=True)
                pattern_q_loss_to_step.backward()
                self.pattern_q_optimizer.step()
                self.pattern_q_optimizer.zero_grad(set_to_none=True)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            if self.eval_freq > 0 and self.step % self.eval_freq == 0:
                self.evaluate()

            if self.step % self.log_freq == 0:
                infos = {**infos, **rollout_info}
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos.items()]
                )
                logger.print(
                    f"{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}"
                )
                metrics = {k: v.detach().item() for k, v in infos.items()}
                logger.log(
                    step=self.step, loss=loss.detach().item(), **metrics, flush=True
                )

            if self.sample_freq and self.step == 0:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                if self.model.__class__ == diffuser.models.diffusion.GaussianDiffusion:
                    self.inv_render_samples()
                elif self.model.__class__ == diffuser.models.diffusion.ValueDiffusion:
                    pass
                else:
                    self.render_samples()

            self.step += 1

    def evaluate(self):
        assert (
            self.evaluator is not None
        ), "Method `evaluate` can not be called when `self.evaluator` is None. Set evaluator with `self.set_evaluator` first."
        self.evaluator.evaluate(load_step=self.step)

    def save(self):
        """
        saves model and ema to disk;
        syncs to storage bucket if a bucket is specified
        """

        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.bucket, logger.prefix, "checkpoint")
        os.makedirs(savepath, exist_ok=True)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f"state_{self.step}.pt")
        else:
            savepath = os.path.join(savepath, "state.pt")
        torch.save(data, savepath)
        logger.print(f"[ utils/training ] Saved model to {savepath}")

    def load(self):
        """
        loads model and ema from disk
        """

        loadpath = os.path.join(self.bucket, logger.prefix, "checkpoint/state.pt")
        data = torch.load(loadpath)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    # -----------------------------------------------------------------------------#
    # --------------------------------- rendering ---------------------------------#
    # -----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        """
        renders training points
        """

        # get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        # get trajectories and condition at t=0 from batch
        trajectories = to_np(batch["x"])
        # conditions = to_np(batch.cond[0])[:, None]

        # [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[..., self.dataset.action_dim :]
        shape = normed_observations.shape
        observations = self.dataset.normalizer.unnormalize(
            normed_observations.reshape(-1, *normed_observations.shape[2:]),
            "observations",
        ).reshape(shape)

        savepath = os.path.join("images", "sample-reference.png")
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        """
        renders samples from (ema) diffusion model
        """
        for i in range(batch_size):
            # get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.cond, self.device)
            player_conditions = None
            if (
                "player_idxs" in conditions and "player_hoop_sides" in conditions
            ):  # must have add player info
                player_conditions = {
                    "player_idxs": conditions["player_idxs"],
                    "player_hoop_sides": conditions["player_hoop_sides"],
                }
                conditions = {
                    key: val
                    for key, val in list(conditions.items())
                    if (key != "player_idxs" and key != "player_hoop_sides")
                }

            # repeat each item in conditions `n_samples` times
            if len(list(conditions.values())[0].shape) == 4:
                conditions = apply_dict(
                    einops.repeat,
                    conditions,
                    "b t a d -> (repeat b) t a d",
                    repeat=n_samples,
                )
            elif len(list(conditions.values())[0].shape) == 3:
                conditions = apply_dict(
                    einops.repeat,
                    conditions,
                    "b a d -> (repeat b) a d",
                    repeat=n_samples,
                )
            else:
                conditions = apply_dict(
                    einops.repeat,
                    conditions,
                    "b d -> (repeat b) d",
                    repeat=n_samples,
                )

            if player_conditions is not None:
                player_conditions = apply_dict(
                    einops.repeat,
                    player_conditions,
                    "b t a d -> (repeat b) t a d",
                    repeat=n_samples,
                )
                for key, val in list(player_conditions.items()):
                    assert key == "player_idxs" or key == "player_hoop_sides"
                    conditions[key] = val

            # [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(
                    torch.ones(n_samples, 1, self.model.n_agents), self.device
                )
            else:
                returns = None

            samples = self.ema_model.conditional_sample(conditions, returns=returns)
            samples = to_np(samples)

            # [ n_samples x horizon x agent x observation_dim ]
            normed_observations = samples[:, :, :, self.dataset.action_dim :]

            # [ 1 x 1 x agent x observation_dim ]
            # normed_conditions = to_np(batch.cond[0])[:, None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            # [ n_samples x (horizon + 1) x agent x observation_dim ]
            # normed_observations = np.concatenate(
            #     [np.repeat(normed_conditions, n_samples, axis=0), normed_observations],
            #     axis=1,
            # )

            # [ n_samples x (horizon + 1) x agent x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(
                normed_observations, "observations"
            )

            # @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join("images", f"sample-{i}.png")
            self.renderer.composite(savepath, observations)

    def inv_render_samples(self, batch_size=2, n_samples=2):
        """
        renders samples from (ema) diffusion model
        """
        for i in range(batch_size):
            # get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch["cond"], self.device)
            # repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                "b ... -> (repeat b) ...",
                repeat=n_samples,
            )
            # [ n_samples x horizon x n_agents x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(
                    torch.ones(n_samples, 1, self.model.n_agents), self.device
                )
            else:
                returns = None

            samples = self.ema_model.conditional_sample(conditions, returns=returns)
            samples = to_np(samples)

            # [ n_samples x horizon x n_agents x observation_dim ]
            normed_observations = samples[:, :, :, :]

            # [ 1 x 1 x n_agents x observation_dim ]
            # normed_conditions = to_np(batch.cond[0])[:, None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            # [ n_samples x (horizon + 1) x n_agents x observation_dim ]
            # normed_observations = np.concatenate(
            #     [np.repeat(normed_conditions, n_samples, axis=0), normed_observations],
            #     axis=1,
            # )

            # [ n_samples x (horizon + 1) x n_agents x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(
                normed_observations, "observations"
            )

            savepath = os.path.join("images", f"sample-{i}.png")
            self.renderer.composite(savepath, observations)
