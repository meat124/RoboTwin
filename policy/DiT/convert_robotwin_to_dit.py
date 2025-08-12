#!/usr/bin/env python3
import argparse
import os
import json
import pickle
from pathlib import Path
import glob

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# ✅ 모든 이미지를 통일할 고정된 크기
IMAGE_SIZE = (256, 256)

def _resize_and_encode(bgr_img: np.ndarray, size=IMAGE_SIZE) -> bytes:
    """
    이미지를 지정된 고정 크기로 리사이즈하고 JPEG로 인코딩합니다.
    """
    # 이미지를 IMAGE_SIZE로 리사이즈
    resized = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
    
    # JPEG로 인코딩하여 바이트로 변환
    _, enc_img = cv2.imencode(".jpg", resized)
    return enc_img.tobytes()

def convert_robotwin_to_dit_format(data_dir: Path, out_dir: Path, gaussian_norm: bool):
    """
    RoboTwin HDF5 파일들을 DiT가 요구하는 단순 pkl 포맷으로 변환합니다.
    """
    hdf5_paths = sorted(glob.glob(str(data_dir / "episode*.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No 'episode*.hdf5' files found in {data_dir}")

    os.makedirs(out_dir, exist_ok=True)

    all_trajs = []
    all_actions = []
    # 카메라 이름 순서를 명시적으로 고정
    camera_names = ['front_camera', 'head_camera', 'left_camera', 'right_camera']

    for hdf5_path in tqdm(hdf5_paths, desc="Converting HDF5 files"):
        try:
            with h5py.File(hdf5_path, "r") as f:
                actions = f["joint_action/vector"][:]
                states = actions
                img_data = {cam: f[f"observation/{cam}/rgb"][:] for cam in camera_names}
                
                T = actions.shape[0]
                current_traj = []
                for t in range(T):
                    obs_dict = {"state": states[t].astype(np.float32)}
                    
                    for i, cam_name in enumerate(camera_names):
                        img_bytes = img_data[cam_name][t]
                        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        # ✅ 리사이즈 및 인코딩 함수 호출
                        obs_dict[f'enc_cam_{i}'] = _resize_and_encode(img_bgr)
                    
                    action = actions[t].astype(np.float32)
                    reward = 0.0
                    current_traj.append((obs_dict, action, reward))
                    all_actions.append(action)

                all_trajs.append(current_traj)

        except Exception as e:
            print(f"Skipping {os.path.basename(str(hdf5_path))} due to an error: {e}")
            continue

    if not all_actions:
        raise RuntimeError("No actions were collected. Check your HDF5 file paths and their contents.")

    # 행동 정규화
    arr = np.array(all_actions)
    if gaussian_norm:
        loc, scale = arr.mean(axis=0), arr.std(axis=0)
    else:
        mx, mn = arr.max(axis=0), arr.min(axis=0)
        loc, scale = (mx + mn) / 2, (mx - mn) / 2
    scale[scale == 0] = 1e-8
    norm = {"loc": loc.tolist(), "scale": scale.tolist()}

    # 최종 파일 저장
    with open(out_dir / "ac_norm.json", "w") as f:
        json.dump(norm, f, indent=4)
    
    with open(out_dir / "buf.pkl", "wb") as f:
        pickle.dump(all_trajs, f)

    print(f"✅ Converted {len(all_trajs)} episodes and saved to '{out_dir}' (resized to {IMAGE_SIZE}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboTwin HDF5 to DiT PKL format.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing episode*.hdf5 files.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for buf.pkl and ac_norm.json.")
    parser.add_argument("--gaussian_norm", action="store_true", help="Use mean/std normalization.")
    args = parser.parse_args()
    
    convert_robotwin_to_dit_format(args.data_dir, args.out_dir, args.gaussian_norm)