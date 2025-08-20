import os
import h5py
import shutil
import argparse
import pickle
import io
from PIL import Image
import re
import numpy as np
import imageio


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


def visualize_state_in_pkl(pkl_path, cam_name='enc_cam_0'):
    """
    Visualizes the 'state' from each timestep in the first episode of a pickle file,
    plots each joint state over time, and saves a video for the specified camera view.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list) or not data:
        print("Pickle file does not contain a list of episodes or is empty.")
        return

    print(f"Visualizing state and creating video for Episode 0 in {pkl_path}:")
    
    episode_data = data[0]
    if not isinstance(episode_data, list):
        print("  Episode 0 is not a list of timesteps.")
        return

    states_over_time = []
    video_frames = []
    for timestep_idx, timestep_tuple in enumerate(episode_data):
        try:
            # Assuming the structure is: list -> list -> tuple -> dict
            # and the dictionary is the first element of the tuple.
            obs_dict = timestep_tuple[0]
            if isinstance(obs_dict, dict):
                if 'state' in obs_dict:
                    state = obs_dict['state']
                    states_over_time.append(state)
                
                # Extract image for video
                if cam_name in obs_dict:
                    # The image is a flattened numpy array, reshape it to (H, W, C)
                    img_array = obs_dict[cam_name]
                    # Assuming image dimensions are 128x128 with 3 channels (RGB)
                    h, w, c = 256, 256, 3
                    if img_array.size == h * w * c:
                        img_array = img_array.reshape(h, w, c)
                        video_frames.append(img_array)
                    else:
                        print(f"  Timestep {timestep_idx}: Image size mismatch for '{cam_name}'. Expected {h*w*c}, got {img_array.size}.")
                else:
                    print(f"  Timestep {timestep_idx}: '{cam_name}' image not found in observation dict.")

            else:
                print(f"  Timestep {timestep_idx}: First element is not a dictionary.")
        except (IndexError, TypeError) as e:
            print(f"  Timestep {timestep_idx}: Could not access data. Unexpected structure: {e}")

    # --- Plotting Joint States ---
    if not states_over_time:
        print("No states found to plot.")
    else:
        states_array = np.array(states_over_time)
        num_timesteps, num_joints = states_array.shape
        timesteps = np.arange(num_timesteps)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        for i in range(num_joints):
            plt.plot(timesteps, states_array[:, i], label=f'Joint {i}')

        plt.title('Joint States Over Time for Episode 0')
        plt.xlabel('Timestep')
        plt.ylabel('Joint State Value')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        
        data_dir = os.path.dirname(pkl_path) or "."
        base_filename = os.path.splitext(os.path.basename(pkl_path))[0]
        plot_filename = f"{base_filename}_joint_states.png"
        save_path = os.path.join(data_dir, plot_filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()

    # --- Saving Video ---
    if not video_frames:
        print(f"No frames found for '{cam_name}' to create a video.")
        return

    data_dir = os.path.dirname(pkl_path) or "."
    base_filename = os.path.splitext(os.path.basename(pkl_path))[0]
    video_filename = f"{base_filename}_{cam_name}_episode0.mp4"
    video_path = os.path.join(data_dir, video_filename)
    
    print(f"Saving video to {video_path}...")
    imageio.mimsave(video_path, video_frames, fps=30, macro_block_size=1)
    print("Video saved successfully.")



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

    # visualize_pkl(args.pkl_path, args.max_depth)
    visualize_state_in_pkl(args.pkl_path)
    # visualize_hdf5(args.hdf5_path)