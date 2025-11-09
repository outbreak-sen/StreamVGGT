import os
import glob
import torch
import numpy as np
import sys
from tqdm import tqdm
from datetime import datetime
sys.path.append("src/")
from visual_util import predictions_to_glb
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import unproject_depth_map_to_point_map
import open3d as o3d


def load_model(ckpt_path="ckpt/checkpoints.pth"):
    print("Loading StreamVGGT model...")
    model = StreamVGGT()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("lch01/StreamVGGT", "checkpoints.pth")
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    del ckpt
    return model


def select_images(image_folder, max_frames=100, sample_interval=1):
    image_list = sorted(glob.glob(os.path.join(image_folder, "*.[jp][pn]g")))
    if len(image_list) == 0:
        raise ValueError(f"No images found in {image_folder}")

    image_list = image_list[::sample_interval]
    if len(image_list) > max_frames:
        step = len(image_list) // max_frames
        image_list = image_list[::step]

    print(f"Using {len(image_list)} images after sampling")
    return image_list


def run_inference(model, image_list, device="cuda"):
    model = model.to(device)
    images = load_and_preprocess_images(image_list).to(device)
    frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]

    print("Running StreamVGGT inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            output = model.inference(frames)

    all_pts3d, all_conf, all_depth, all_pose = [], [], [], []
    for res in output.ress:
        all_pts3d.append(res["pts3d_in_other_view"].squeeze(0).cpu())
        all_conf.append(res["conf"].squeeze(0).cpu())
        all_depth.append(res["depth"].squeeze(0).cpu())
        all_pose.append(res["camera_pose"].squeeze(0).cpu())

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        torch.stack(all_pose, dim=0).unsqueeze(0),
        images.shape[-2:]
    )

    predictions = {
        "world_points": torch.stack(all_pts3d, dim=0).numpy(),
        "world_points_conf": torch.stack(all_conf, dim=0).numpy(),
        "depth": torch.stack(all_depth, dim=0).numpy(),
        "extrinsic": extrinsic.squeeze(0).numpy(),
        "intrinsic": intrinsic.squeeze(0).numpy() if intrinsic is not None else None
    }
    return predictions


def save_ply(predictions, out_path="output/scene.ply", conf_thres=0.5):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pts3d = predictions["world_points"]
    conf = predictions["world_points_conf"]

    mask = conf > conf_thres
    pts = pts3d[mask]
    print(f"Saving {pts.shape[0]} valid points to {out_path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(out_path, pcd)


def save_tum_traj(predictions, image_list, out_path="output/traj.txt"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    extrinsics = predictions["extrinsic"]  # (N, 3, 4)
    with open(out_path, "w") as f:
        f.write("# ground truth trajectory\n")
        f.write(f"# file: '{out_path[-1]}'\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i, img_path in enumerate(image_list):
            name = os.path.splitext(os.path.basename(img_path))[0]
            R = extrinsics[i][:, :3]
            t = extrinsics[i][:, 3]
            from scipy.spatial.transform import Rotation as R_
            q = R_.from_matrix(R).as_quat()  # [x, y, z, w]
            f.write(f"{name} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    print(f"TUM trajectory saved to {out_path}")


def main(
    image_folder,
    output_dir="output",
    max_frames=200,
    sample_interval=2,
    conf_thres=3.0,
    args=None
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model()
    image_list = select_images(image_folder, max_frames, sample_interval)
    preds = run_inference(model, image_list, device)
    image_folder = os.path.abspath(args.image_folder)
    if args.tum:
        # TUM 模式：取上一级文件夹名称
        dataset_name = os.path.basename(os.path.dirname(image_folder))
        ply_name = f"{dataset_name}.ply"
        traj_name = f"traj_{dataset_name}.txt"
    else:
        # 非 TUM 模式：取最后一层文件夹名称
        folder_name = os.path.basename(image_folder)
        ply_name = f"{folder_name}.ply"
        traj_name = f"traj_{folder_name}.txt"

    ply_path = os.path.join(args.output_dir, ply_name)
    traj_path = os.path.join(args.output_dir, traj_name)
    
    print(f"[INFO] Saving results to {ply_path} and {traj_path}")
    save_ply(preds,ply_path, conf_thres)
    save_tum_traj(preds, image_list, traj_path)
    print("✅ Reconstruction completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="StreamVGGT image folder reconstruction")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--conf_thres", type=float, default=3.0)
    parser.add_argument("--tum", action="store_true", help="Use TUM dataset naming convention")

    args = parser.parse_args()

    main(
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        sample_interval=args.sample_interval,
        conf_thres=args.conf_thres,
        args=args
    )
