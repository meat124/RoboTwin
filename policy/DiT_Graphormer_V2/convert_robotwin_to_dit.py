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

def _encode_depth(depth_map: np.ndarray) -> bytes:
    """
    정규화된 [0, 1] 범위의 float Depth 맵을 16비트 정수형으로 변환하고
    무손실 PNG로 인코딩합니다.
    """
    # [0, 1] 범위를 [0, 65535] 범위의 16비트 정수(uint16)로 변환
    depth_uint16 = (depth_map * 65535).astype(np.uint16)
    
    # PNG로 인코딩
    _, enc_img = cv2.imencode(".png", depth_uint16)
    return enc_img.tobytes()

def _encode_normal(normal_map: np.ndarray) -> bytes:
    """
    정규화된 [0, 1] 범위의 float Normal 맵을 8비트 정수형으로 변환하고
    무손실 PNG로 인코딩합니다.
    """
    # [0, 1] 범위를 [0, 255] 범위의 8비트 정수(uint8)로 변환
    normal_uint8 = (normal_map * 255).astype(np.uint8)
    
    # PNG로 인코딩
    _, enc_img = cv2.imencode(".png", normal_uint8)
    return enc_img.tobytes()

def _encode_mask(mask_map: np.ndarray) -> bytes:
    """
    uint8 Mask 맵을 무손실 PNG로 인코딩합니다.
    """
    _, enc_img = cv2.imencode(".png", mask_map)
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


# def convert_robotwin_to_dit_depth_format(data_dir: Path, out_dir: Path, gaussian_norm: bool):
#     """
#     RoboTwin DiT_Graphormer_V2 + depth
#     """
#     import cv2
#     import torch
#     from MoGe.moge.model.v2 import MoGeModel

#     device = torch.device("cuda")
#     model = MoGeModel.from_pretrained("/scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT_Graphormer_V2/visual_features/moge_model/model_vitl_normal.pt").to(device)

#     hdf5_paths = sorted(glob.glob(str(data_dir / "episode*.hdf5")))
#     if not hdf5_paths:
#         raise FileNotFoundError(f"No 'episode*.hdf5' files found in {data_dir}")

#     os.makedirs(out_dir, exist_ok=True)

#     all_trajs = []
#     all_actions = []
#     # 카메라 이름 순서를 명시적으로 고정
#     # camera_names = ['front_camera', 'head_camera', 'left_camera', 'right_camera']
#     camera_names = ['head_camera']  # DiT_Graphormer_V2 MoGe uses only head camera

#     for hdf5_path in tqdm(hdf5_paths, desc="Converting HDF5 files"):
#         try:
#             with h5py.File(hdf5_path, "r") as f:
#                 actions = f["joint_action/vector"][:]
#                 states = actions
#                 img_data = {cam: f[f"observation/{cam}/rgb"][:] for cam in camera_names}
                
#                 T = actions.shape[0]
#                 current_traj = []
#                 for t in range(T):
#                     print(f"Processing timestep {t}/{T}", end='\r')
#                     obs_dict = {"state": states[t].astype(np.float32)}
                    
#                     for i, cam_name in enumerate(camera_names):
#                         img_bytes = img_data[cam_name][t]
#                         img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
#                         # ✅ 리사이즈 및 인코딩 함수 호출
#                         obs_dict[f'enc_cam_{i}'] = _resize_and_encode(img_bgr)
                        
#                         # MoGe
#                         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#                         resized = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
#                         resized = torch.tensor(resized / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
#                         with torch.no_grad():
#                             output = model.infer(resized)
                        
#                         depth = output.get("depth").cpu().numpy()  # (H, W)
#                         normal = output.get("normal").cpu().numpy()  # (H, W, 3)
#                         mask = output.get("mask").cpu().numpy()  # (H, W)

#                         obs_dict[f'cam_{i}_depth'] = depth.astype(np.float16)
#                         obs_dict[f'cam_{i}_normal'] = normal.astype(np.float16)
#                         obs_dict[f'cam_{i}_mask'] = mask.astype(np.uint8)

#                     action = actions[t].astype(np.float32)
#                     reward = 0.0
#                     current_traj.append((obs_dict, action, reward))
#                     all_actions.append(action)

#                 all_trajs.append(current_traj)

#         except Exception as e:
#             print(f"Skipping {os.path.basename(str(hdf5_path))} due to an error: {e}")
#             continue

#     if not all_actions:
#         raise RuntimeError("No actions were collected. Check your HDF5 file paths and their contents.")

#     # 행동 정규화
#     arr = np.array(all_actions)
#     if gaussian_norm:
#         loc, scale = arr.mean(axis=0), arr.std(axis=0)
#     else:
#         mx, mn = arr.max(axis=0), arr.min(axis=0)
#         loc, scale = (mx + mn) / 2, (mx - mn) / 2
#     scale[scale == 0] = 1e-8
#     norm = {"loc": loc.tolist(), "scale": scale.tolist()}

#     # 최종 파일 저장
#     with open(out_dir / "ac_norm.json", "w") as f:
#         json.dump(norm, f, indent=4)
    
#     with open(out_dir / "buf_moge.pkl", "wb") as f:
#         pickle.dump(all_trajs, f)

#     print(f"✅ Converted {len(all_trajs)} episodes and saved to '{out_dir}' (resized to {IMAGE_SIZE}).")


# def normalize_depth_and_normal(data_dir: Path):
#     """
#     buf_moge.pkl 파일을 로드하여 depth와 normal 데이터의 전체 최솟값/최댓값을 계산하고 정규화합니다.
#     - Depth: inf 값을 1.0으로 처리
#     - Normal: inf 값이 없음을 가정하고 처리
#     """
#     input_path = data_dir / "buf_moge.pkl"
#     output_path = data_dir / "buf_moge_normalized.pkl"
    
#     print(f"'{input_path}'에서 데이터 로딩 중...")
#     with open(input_path, "rb") as f:
#         data = pickle.load(f)

#     # --- 1단계: 전체 데이터셋을 순회하며 최솟값/최댓값 찾기 ---
#     print("1단계: 최솟값/최댓값 계산 중...")
#     depth_min, depth_max = np.inf, -np.inf
#     normal_min, normal_max = np.inf, -np.inf

#     for traj in tqdm(data, desc="Finding Min/Max"):
#         for obs, _, _ in traj:
#             if "cam_0_mask" in obs:
#                 mask = obs["cam_0_mask"].astype(bool)

#                 # Depth는 inf 값이 있으므로, mask와 isfinite를 모두 사용
#                 if "cam_0_depth" in obs:
#                     depth_map = obs["cam_0_depth"].astype(np.float32)
#                     combined_mask = mask & np.isfinite(depth_map)
#                     valid_depths = depth_map[combined_mask]
#                     if valid_depths.size > 0:
#                         depth_min = min(depth_min, np.min(valid_depths))
#                         depth_max = max(depth_max, np.max(valid_depths))
                
#                 # ✅ [수정] Normal은 inf 값이 없으므로, mask만 사용
#                 if "cam_0_normal" in obs:
#                     normal_map = obs["cam_0_normal"].astype(np.float32)
#                     valid_normals = normal_map[mask]
#                     if valid_normals.size > 0:
#                         normal_min = min(normal_min, np.min(valid_normals))
#                         normal_max = max(normal_max, np.max(valid_normals))

#     print("\n계산 완료!")
#     print(f"  - Global Finite Depth Range (Masked):  [{depth_min:.4f}, {depth_max:.4f}]")
#     print(f"  - Global Normal Range (Masked): [{normal_min:.4f}, {normal_max:.4f}]")

#     # --- 2단계: 계산된 최솟값/최댓값으로 데이터 정규화 ---
#     print("\n2단계: 데이터 정규화 중...")
    
#     epsilon = 1e-8
#     depth_range = depth_max - depth_min
#     normal_range = normal_max - normal_min

#     for traj in tqdm(data, desc="Normalizing Data"):
#         for obs, _, _ in traj:
#             # Depth는 inf 처리가 필요하므로 복잡한 로직 유지
#             if "cam_0_depth" in obs:
#                 depth_map = obs["cam_0_depth"].astype(np.float32)
#                 finite_mask = np.isfinite(depth_map)
#                 normalized_map = np.full_like(depth_map, 1.0, dtype=np.float32)
                
#                 valid_depths = depth_map[finite_mask]
#                 normalized_values = (valid_depths - depth_min) / (depth_range + epsilon)
                
#                 normalized_map[finite_mask] = normalized_values
#                 obs["cam_0_depth"] = normalized_map.astype(np.float16)

#             # ✅ [수정] Normal은 inf가 없으므로, 간단한 정규화 수식을 전체에 적용
#             if "cam_0_normal" in obs:
#                 normal_map = obs["cam_0_normal"].astype(np.float32)
#                 normalized_normal = (normal_map - normal_min) / (normal_range + epsilon)
#                 obs["cam_0_normal"] = normalized_normal.astype(np.float16)

#     print(f"\n정규화 완료. '{output_path}'에 저장 중...")
#     with open(output_path, "wb") as f:
#         pickle.dump(data, f)
    
#     print(f"✅ 정규화된 데이터가 성공적으로 저장되었습니다.")


def convert_robotwin_to_dit_moge_format(data_dir: Path, out_dir: Path, gaussian_norm: bool):
    """
    [통합 버전] 데이터 변환, 통계 계산, 정규화, 인코딩을 한 번에 효율적으로 처리합니다.
    model.infer는 단 한 번만 호출됩니다.
    """
    import cv2
    import torch
    from MoGe.moge.model.v2 import MoGeModel

    device = torch.device("cuda")
    model = MoGeModel.from_pretrained("/scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT_Graphormer_V2/visual_features/moge_model/model_vitl_normal.pt").to(device)

    hdf5_paths = sorted(glob.glob(str(data_dir / "episode*.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No 'episode*.hdf5' files found in {data_dir}")

    os.makedirs(out_dir, exist_ok=True)
    camera_names = ['head_camera']

    # --- 1단계: 데이터 생성 및 통계 동시 수집 ---
    print("1단계: 데이터 생성 및 통계 수집 중 (model.infer는 여기서 한 번만 실행)...")
    all_trajs = []
    all_actions = []
    depth_min, depth_max = np.inf, -np.inf
    normal_min, normal_max = np.inf, -np.inf

    for hdf5_path in tqdm(hdf5_paths, desc="Generating data & stats"):
        with h5py.File(hdf5_path, "r") as f:
            actions = f["joint_action/vector"][:]
            states = actions
            img_data = {cam: f[f"observation/{cam}/rgb"][:] for cam in camera_names}
            T = actions.shape[0]
            current_traj = []

            for t in range(T):
                obs_dict = {"state": states[t].astype(np.float32)}
                
                img_bytes = img_data[camera_names[0]][t]
                img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                obs_dict['enc_cam_0'] = _resize_and_encode(img_bgr) # RGB는 바로 인코딩
                
                # MoGe 추론 (가장 비싼 연산)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                resized = torch.tensor(resized / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
                with torch.no_grad():
                    output = model.infer(resized)
                
                depth = output.get("depth").cpu().numpy()
                normal = output.get("normal").cpu().numpy()
                mask = output.get("mask").cpu().numpy()

                # ✅ [변경] obs_dict에는 원본(raw) 배열을 그대로 저장
                obs_dict['cam_0_depth'] = depth
                obs_dict['cam_0_normal'] = normal
                obs_dict['cam_0_mask'] = mask.astype(np.uint8)

                # ✅ [변경] 저장과 동시에 통계 갱신
                mask_bool = mask.astype(bool)
                combined_mask = mask_bool & np.isfinite(depth)
                valid_depths = depth[combined_mask]
                if valid_depths.size > 0:
                    depth_min = min(depth_min, np.min(valid_depths))
                    depth_max = max(depth_max, np.max(valid_depths))
                
                valid_normals = normal[mask_bool]
                if valid_normals.size > 0:
                    normal_min = min(normal_min, np.min(valid_normals))
                    normal_max = max(normal_max, np.max(valid_normals))

                action = actions[t].astype(np.float32)
                reward = 0.0
                current_traj.append((obs_dict, action, reward))
                all_actions.append(action)
            all_trajs.append(current_traj)

    print("\n통계 계산 완료!")
    print(f"  - Global Depth Range:  [{depth_min:.4f}, {depth_max:.4f}]")
    print(f"  - Global Normal Range: [{normal_min:.4f}, {normal_max:.4f}]")
    
    # --- 2단계: 후처리 (정규화 및 인코딩) ---
    print("\n2단계: 후처리(정규화 및 인코딩) 시작...")
    epsilon = 1e-8
    depth_range = depth_max - depth_min
    normal_range = normal_max - normal_min

    for traj in tqdm(all_trajs, desc="Post-processing & Encoding"):
        for obs, _, _ in traj:
            # Depth 정규화 및 인코딩
            raw_depth = obs.pop('cam_0_depth') # 원본 배열 가져오고 dict에서 삭제
            finite_mask = np.isfinite(raw_depth)
            normalized_depth = np.full_like(raw_depth, 1.0, dtype=np.float32)
            valid_depths = raw_depth[finite_mask]
            normalized_values = (valid_depths - depth_min) / (depth_range + epsilon)
            normalized_depth[finite_mask] = normalized_values
            obs['enc_cam_0_depth'] = _encode_depth(normalized_depth) # 인코딩된 값 추가

            # Normal 정규화 및 인코딩
            raw_normal = obs.pop('cam_0_normal')
            normalized_normal = (raw_normal - normal_min) / (normal_range + epsilon)
            obs['enc_cam_0_normal'] = _encode_normal(normalized_normal)

            # Mask 인코딩
            raw_mask = obs.pop('cam_0_mask')
            obs['enc_cam_0_mask'] = _encode_mask(raw_mask)
    
    print("\n후처리 완료.")

    # ... (행동 정규화 및 최종 파일 저장 부분은 동일) ...
    arr = np.array(all_actions)
    if gaussian_norm:
        loc, scale = arr.mean(axis=0), arr.std(axis=0)
    else:
        mx, mn = arr.max(axis=0), arr.min(axis=0)
        loc, scale = (mx + mn) / 2, (mx - mn) / 2
    scale[scale == 0] = 1e-8
    norm = {"loc": loc.tolist(), "scale": scale.tolist()}

    # with open(out_dir / "ac_norm.json", "w") as f:
    #     json.dump(norm, f, indent=4)
    
    with open(out_dir / "buf_moge_enc_normalized.pkl", "wb") as f:
        pickle.dump(all_trajs, f)

    print(f"✅ Converted {len(all_trajs)} episodes and saved to '{out_dir}' (resized to {IMAGE_SIZE}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboTwin HDF5 to DiT_Graphormer_V2 PKL format.")
    parser.add_argument("--data_dir", type=Path, required=False, help="Directory containing episode*.hdf5 files.")
    parser.add_argument("--out_dir", type=Path, required=False, help="Output directory for buf.pkl and ac_norm.json.")
    parser.add_argument("--gaussian_norm", action="store_true", help="Use mean/std normalization.")
    args = parser.parse_args()
    
    # convert_robotwin_to_dit_format(args.data_dir, args.out_dir, args.gaussian_norm)
    # convert_robotwin_to_dit_depth_format(args.data_dir, args.out_dir, args.gaussian_norm)
    # normalize_depth_and_normal(args.data_dir)
    convert_robotwin_to_dit_moge_format(args.data_dir, args.out_dir, args.gaussian_norm)