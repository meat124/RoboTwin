import os
import h5py
import shutil
import argparse
import pickle
import io
from PIL import Image
import re
import numpy as np


def print_structure(name, obj, indent=0):
    prefix = " " * indent
    if isinstance(obj, h5py.Group):
        print(f"{prefix}- {name}/ (Group)")
        for key, val in obj.items():
            print_structure(key, val, indent + 4)
    elif isinstance(obj, h5py.Dataset):
        print(f"{prefix}- {name} (Dataset, shape={obj.shape}, dtype={obj.dtype})")
    else:
        print(f"{prefix}- {name} (Unknown)")


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector, image_dict


def visualize_hdf5(hdf5_path=None, subpath=None, view_camera=None, frame_idx=0):
    print(f"Visualizing HDF5 structure for {hdf5_path}...")
    with h5py.File(hdf5_path, "r") as f:
        if subpath:
            # ensure it exists
            if subpath not in f and subpath not in f.keys():
                raise KeyError(f"Group '{subpath}' not found in file.")
            group = f[subpath]
            print(f"Structure of '{hdf5_path}' -> '{subpath}':")
            print_structure(subpath, group, indent=0)
        else:
            print(f"Structure of '{hdf5_path}':")
            for key, val in f.items():
                print_structure(key, val, indent=0)

        # if a camera is specified, visualize it
        if view_camera:
            # validate camera
            obs_group = f["observation"]
            if view_camera not in obs_group:
                raise KeyError(f"Camera '{view_camera}' not found in /observation")
            rgb_ds = obs_group[view_camera]["rgb"]
            all_images = rgb_ds[()]
            if frame_idx < 0 or frame_idx >= len(all_images):
                raise IndexError(f"frame_idx {frame_idx} out of range (0..{len(all_images)-1})")

            # extract raw bytes and decode
            import matplotlib.pyplot as plt

            raw = all_images[frame_idx]
            img = Image.open(io.BytesIO(raw.tobytes()))
            plt.imshow(img)
            plt.title(f"{view_camera} frame {frame_idx}")
            plt.axis("off")
            plt.show()


def visualize_pkl(pkl_path, max_depth):
    """
    Visualizes the structure of a pickle file, showing only the first
    element of lists/tuples to keep the output concise.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    def print_pkl_structure(obj, name="root", indent=0, depth=0):
        if max_depth is not None and depth > max_depth:
            print(f"{' ' * indent}{name} ... (max depth reached)")
            return
        
        prefix = ' ' * indent
        t = type(obj)

        if isinstance(obj, dict):
            print(f"{prefix}{name}/ (dict, keys={list(obj.keys())})")
            for k, v in obj.items():
                print_pkl_structure(v, name=str(k), indent=indent + 4, depth=depth + 1)
        elif isinstance(obj, (list, tuple)):
            length = len(obj)
            print(f"{prefix}{name} ({t.__name__}, len={length})")
            if length > 0:
                # Only inspect the first element to represent the structure
                print_pkl_structure(obj[0], name="[0] (sample element)", indent=indent + 4, depth=depth + 1)
        elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
            # Numpy arrays, tensors, etc.
            print(f"{prefix}{name} (array, shape={obj.shape}, dtype={obj.dtype})")
        else:
            # Primitive types or other objects
            print(f"{prefix}{name} ({t.__name__})")

    print(f"Structure of '{pkl_path}' (showing first element of lists as sample):")
    print_pkl_structure(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 to PKL conversion or visualization")
    parser.add_argument(
        "--pkl_path",
        required=False,
        help="Path to the .pkl file"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum recursion depth for nested structures"
    )
    parser.add_argument(
        "--hdf5_path",
        required=False,
        help="Path to the HDF5 file for visualization"
    )
    args = parser.parse_args()

    visualize_pkl(args.pkl_path, args.max_depth)
    # visualize_hdf5(args.hdf5_path)