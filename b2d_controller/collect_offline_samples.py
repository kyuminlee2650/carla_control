r"""Collect real VAD trajectory predictions OFFLINE, from Bench2DriveZoo's own saved val dataset
(data/bench2drive/v1/ + data/infos/b2d_infos_val.pkl) -- no CARLA server, no live rendering, so none
of the GPU contention that made online (leaderboard_evaluator.py) collection OOM after 1-2 ticks on
this machine's 10GB card (CARLA's own headless rendering alone was using ~8GB regardless of map
size). Pure PyTorch forward passes over saved images instead: the whole 10GB is available to the
model, and there's no per-tick world/scenario setup overhead, so collecting O(100) samples here
costs a small fraction of what a single CARLA route did.

Run inside the b2d_zoo conda env, from anywhere (this script cd's into Bench2DriveZoo itself,
since the config's data_root/ann_file paths are relative to that directory):
    conda activate b2d_zoo
    python3 collect_offline_samples.py --n 100

Output: a pickle of one dict per sample (same schema mpc_kf_controller.py's build_trajectory_splines
consumes): out_truck (VAD's own [lateral, forward] waypoints), target (approximated, see below),
speed, angular_velocity, acceleration, plus folder/frame_idx for traceability back to the dataset.

target approximation: the live agent's own local_command_xy comes from a route planner's near-term
command point, transformed into the ego frame via a specific (compass-based, not ego_yaw-based)
sign convention baked into vad_b2d_agent.py's tick()/run_step() -- reproducing that exactly from
this dataset's own (differently-conventioned) ego_yaw/world2lidar fields would need reverse-
engineering a frame convention this script has no independent way to verify. Since `target` is only
ever used as one extra shape point stabilizing the curvature spline's tail (not load-bearing for
validating the core out_truck-driven curvature fit), it's approximated instead as a linear
extrapolation of out_truck's own last segment -- geometrically reasonable, and it sidesteps that
whole frame-convention risk entirely rather than risk silently feeding the fit a wrong-signed point.
"""
import argparse
import os
import pickle
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="number of val samples to run (max 442)")
    parser.add_argument("--start", type=int, default=0, help="starting index into the val set")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "vad_trajectory_samples_offline.pkl"))
    parser.add_argument("--bench2drivezoo", default="/home/ailab/project/Bench2DriveZoo")
    parser.add_argument("--ckpt", default="/home/ailab/project/Bench2DriveZoo/ckpts/vad_b2d_base.pth")
    parser.add_argument("--config", default="adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py",
                        help="relative to --bench2drivezoo")
    args = parser.parse_args()

    os.chdir(args.bench2drivezoo)   # data_root/ann_file in the config are relative to here
    sys.path.insert(0, args.bench2drivezoo)

    import torch
    from mmcv import Config
    from mmcv.models import build_model
    from mmcv.utils import load_checkpoint
    from mmcv.datasets import build_dataset
    from mmcv.parallel.collate import collate as mm_collate_to_batch_form

    cfg = Config.fromfile(args.config)
    if getattr(cfg, "plugin", False):
        import importlib
        plugin_dir = os.path.join("Bench2DriveZoo", cfg.plugin_dir)
        module_path = ".".join(os.path.dirname(plugin_dir).split("/"))
        importlib.import_module(module_path)

    print("Building model...")
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=True)
    model.cuda().eval()

    # Same pipeline the live agent uses (inference_only_pipeline: img + ego_fut_cmd only, no GT
    # loading) rather than the heavier GT-collecting test_pipeline cfg.data.test defaults to --
    # keeps this script's forward pass identical to what vad_b2d_agent.py actually runs online.
    test_cfg = dict(cfg.data.test)
    test_cfg["pipeline"] = cfg.inference_only_pipeline
    dataset = build_dataset(test_cfg)
    print(f"Dataset size: {len(dataset)}")

    end = min(args.start + args.n, len(dataset))
    indices = list(range(args.start, end))
    n = len(indices)
    records = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            raw_info = dataset.data_infos[idx]
            results = dataset[idx]
            batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
            for key, data in batch.items():
                if key != "img_metas" and torch.is_tensor(data[0]):
                    data[0] = data[0].cuda()

            out = model(batch, return_loss=False, rescale=True)
            fut_preds = out[0]["pts_bbox"]["ego_fut_preds"].cpu().numpy()   # (n_cmd, n_p, 2)
            all_out_truck = np.cumsum(fut_preds, axis=1)

            ego_fut_cmd = results["ego_fut_cmd"]
            ego_fut_cmd = ego_fut_cmd.data if hasattr(ego_fut_cmd, "data") else ego_fut_cmd
            command = int(np.argmax(np.asarray(ego_fut_cmd).reshape(-1)))
            out_truck = all_out_truck[command]

            last_seg = out_truck[-1] - out_truck[-2]
            target = out_truck[-1] + last_seg   # see module docstring: approximated, not the real
                                                # route-command point

            speed = float(np.linalg.norm(np.asarray(raw_info["ego_vel"], dtype=float)[:2]))
            record = dict(
                idx=idx, folder=raw_info["folder"], frame_idx=raw_info["frame_idx"],
                out_truck=out_truck, target=target, speed=speed,
                angular_velocity=np.asarray(raw_info["ego_rotation_rate"], dtype=float),
                acceleration=np.asarray(raw_info["ego_accel"], dtype=float),
            )
            records.append(record)
            if i % 10 == 0 or i == n - 1:
                print(f"{i + 1}/{n} (idx={idx})  folder={raw_info['folder']}  speed={speed:.2f} m/s  "
                     f"out_truck[-1]={out_truck[-1]}")

    with open(args.out, "wb") as f:
        pickle.dump(records, f)
    print(f"\nSaved {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
