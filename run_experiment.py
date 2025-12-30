import argparse
import datetime
import os
import sys
from subprocess import Popen
from time import sleep

import dateutil.tz
import yaml

from diffuser.utils.launcher_util import RUN, build_nested_variant_generator

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_config", help="experiment config file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()

    with open(args.exp_config, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.FullLoader)

    # generating the variants
    vg_fn = build_nested_variant_generator(exp_specs)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    variants_log_dir = os.path.join(
        RUN.script_root,
        f"logs/variants/variants-for-{exp_specs['meta_data']['exp_name']}",
        "variants-" + timestamp,
    )
    os.makedirs(variants_log_dir)
    with open(os.path.join(variants_log_dir, "exp_spec_definition.yaml"), "w") as f:
        yaml.dump(exp_specs, f, default_flow_style=False)
    num_variants = 0
    for variant in vg_fn():
        i = num_variants
        variant["exp_id"] = i
        with open(os.path.join(variants_log_dir, "%d.yaml" % i), "w") as f:
            yaml.dump(variant, f, default_flow_style=False)
            f.flush()
        num_variants += 1

    num_workers = min(exp_specs["meta_data"]["num_workers"], num_variants)
    exp_specs["meta_data"]["num_workers"] = num_workers

    # run the processes
    running_processes = []
    args_idx = 0

    command_format_dict = exp_specs["meta_data"]
    python_exec = sys.executable
    # Ensure child processes import *this* repo's modules (not an editable install from elsewhere).
    child_cwd = RUN.script_root
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = (
        RUN.script_root
        + (os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")
    )

    while (args_idx < num_variants) or (len(running_processes) > 0):
        if (len(running_processes) < num_workers) and (args_idx < num_variants):
            command_format_dict["specs"] = os.path.join(
                variants_log_dir, "%i.yaml" % args_idx
            )
            command_format_dict["gpuid"] = args.gpu
            script_path = command_format_dict["script_path"]
            if not os.path.isabs(script_path):
                script_path = os.path.join(RUN.script_root, script_path)
            command_to_run = [
                python_exec,
                script_path,
                "-e",
                command_format_dict["specs"],
                "-g",
                str(command_format_dict["gpuid"]),
            ]
            print(command_to_run)
            p = Popen(command_to_run, cwd=child_cwd, env=child_env)
            args_idx += 1
            running_processes.append(p)
        else:
            sleep(1)

        new_running_processes = []
        for p in running_processes:
            ret_code = p.poll()
            if ret_code is None:
                new_running_processes.append(p)
        running_processes = new_running_processes
