import os
import h5py
import shutil
import argparse
import pickle
import io
from PIL import Image
import re
import numpy as np


def convert_hdf5s_to_pkl(input_dir, output_pkl_path):
    """
    Convert all HDF5 episode files in input_dir into a pickle file where
    each episode is a list of [obs_dict, action_vector, reward_scalar] per timestep.
    """
    # collect and sort .hdf5 files by episode index
    files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    def ep_idx(fn):
        m = re.search(r'episode(\d+)', fn)
        return int(m.group(1)) if m else fn
    files = sorted(files, key=ep_idx)

    episodes_list = []
    for fn in files:
        print(f"Processing {fn}...")
        path = os.path.join(input_dir, fn)
        with h5py.File(path, 'r') as root:
            # load full arrays
            # observations: dict of arrays (T x ...)
            obs_dict = {}
            if 'observation' in root:
                for key, grp in root['observation'].items():
                    if 'rgb' in grp:
                        # This dataset contains compressed images (e.g., JPEG) as byte strings
                        obs_dict[key] = grp['rgb'][()]
                    else:
                        try:
                            obs_dict[key] = grp[()]
                        except Exception:
                            # nested groups
                            nested = {}
                            for k2, v2 in grp.items():
                                nested[k2] = v2[()]
                            obs_dict[key] = nested
            # actions: use 'vector' if present, else combine joint_action keys
            if 'joint_action' in root and 'vector' in root['joint_action']:
                action_arr = root['joint_action']['vector'][()]
            else:
                # combine any joint_action datasets into one vector
                parts = []
                for k, ds in root['joint_action'].items():
                    parts.append(ds[()])
                action_arr = np.concatenate(parts, axis=-1)

            # number of timesteps
            T = action_arr.shape[0]
            episode = []
            for t in range(T):
                # per-timestep observation dict
                step_obs = {}
                for ok, ov in obs_dict.items():
                    # pick t-th slice
                    val = ov[t] if hasattr(ov, 'shape') and len(ov.shape) >= 1 else ov
                    
                    # If the value is a byte string, decode it into an image array
                    if isinstance(val, (bytes, np.bytes_)):
                        try:
                            img = Image.open(io.BytesIO(val))
                            val = np.array(img, dtype=np.uint8) # Convert to numpy array
                        except Exception as e:
                            print(f"Warning: Could not decode image for key {ok} at timestep {t}. Error: {e}")
                    
                    step_obs[ok] = val

                # action vector and reward
                step_action = action_arr[t]
                # convert reward to float if available, else None
                step_reward = float(0.0)  # dummy value
                episode.append([step_obs, step_action, step_reward])
        episodes_list.append(episode)
        print(f"Processed episode {len(episodes_list)}: {fn}")

    # write out to pickle
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(episodes_list, f)
    print(f"Converted {len(episodes_list) - 1} episodes into '{output_pkl_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 to PKL conversion or visualization")
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing episode HDF5 files"
    )
    parser.add_argument(
        "--output_pkl_path", required=True,
        help="Path to save the output pickle file"
    )

    args = parser.parse_args()
    convert_hdf5s_to_pkl(args.input_dir, args.output_pkl_path)
