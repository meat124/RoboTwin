# import packages and module here
import hydra
import torch
import sys, os
import yaml
import cv2
import numpy as np
import json

from .dit_model import DiT
import matplotlib.pyplot as plt
from collections import deque
from torchvision import transforms


IMAGE_SIZE = (256, 256)  # Fixed image size for resizing
EXP_WEIGHT = 0.01
_AC_LOC = None
_AC_SCALE = None
cam_idx_dict = {
    0: "front_camera",
    1: "head_camera",
    2: "left_camera",
    3: "right_camera"
}


def get_preproc_transform(size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size), antialias=False),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


def encode_obs(observation, cam_indexes=[0, 1, 2, 3]):  # Post-Process Observation
    def resize_obs(bgr_img: np.ndarray, size=IMAGE_SIZE):
        resized = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
        resized_rgb = resized[:, :, ::-1].copy()  # Convert BGR to RGB
        # resized = get_preproc_transform(size[0])(resized)
        # resized = resized.float() / 255.0  # Convert to CxHxW format and normalize
        return torch.from_numpy(resized_rgb).permute(2, 0, 1).float() / 255.0  # Convert to CxHxW format and normalize

    obs = {}
    idx = 0
    for cam_idx in cam_indexes:
        cam_name = cam_idx_dict[cam_idx]
        if cam_name in observation["observation"]:
            obs[f"cam{idx}"] = resize_obs(observation["observation"][cam_name]["rgb"])
            idx += 1

    obs["agent_pos"] = torch.tensor(observation["joint_action"]["vector"], dtype=torch.float32)
    obs["agent_pos"] = (obs["agent_pos"] - _AC_LOC) / _AC_SCALE

    return obs


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    # temporally hardcode the path to the model checkpoint
    ckpt_file = usr_args['ckpt_setting']

    load_config_path = f"./policy/DiT/experiments/finetune.yaml"
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    ac_chunk = model_training_config['ac_chunk']
    img_chunk = model_training_config['img_chunk']

    import shutil
    norm_file = os.path.join(os.path.dirname(ckpt_file), "ac_norm.json")
    if os.path.exists(norm_file):
        shutil.copyfile(norm_file, "./ac_norm.json")

    model = DiT(
        ckpt_file=ckpt_file,
        ac_chunk=ac_chunk,
        img_chunk=img_chunk
    )

    return model


act_history = None
action_hist = deque(maxlen=2)
T = 0


def eval(TASK_ENV, model, observation, temporal_ensemble=False, norm=None, cam_indexes=[0, 1, 2, 3]):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    global act_history
    global _AC_LOC, _AC_SCALE
    global T
    _AC_LOC = norm["loc"]
    _AC_SCALE = norm["scale"]

    obs = encode_obs(observation, cam_indexes=cam_indexes)
    instruction = TASK_ENV.get_instruction()
    
    if temporal_ensemble:
        # ======== Get Action with Temporal Ensemble ========
        actions = model.get_action(obs)
        if act_history is None:
            act_history = deque(maxlen=len(actions))
        act_history.append(actions)

        num_actions = len(act_history)
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), act_history
                )
            ]
        )

        # compute the weighted average across all predictions for this timestep
        weights = np.exp(-EXP_WEIGHT * np.arange(num_actions))[::-1]
        weights = weights / weights.sum()
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)  # weighted average

        action = action * _AC_SCALE + _AC_LOC  # denormalize the action

        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation, cam_indexes=cam_indexes)
        model.update_obs(obs)
    else:
        # ======== Get Action without Temporal Ensemble ========
        actions = model.get_action(obs)
        action_hist.append(actions)
        if len(action_hist) == 2:
            prev_action_last_chunk = action_hist[0][-1]
            curr_action_first_chunk = action_hist[1][0]
            # print(f"prev_action_last_chunk.shape: {prev_action_last_chunk.shape}, curr_action_first_chunk.shape: {curr_action_first_chunk.shape}")
            cartesian_prev_action_last_chunk = get_cartesian_pose(prev_action_last_chunk)
            cartesian_curr_action_first_chunk = get_cartesian_pose(curr_action_first_chunk)
            # print(f"cartesian_prev_action_last_chunk.shape: {cartesian_prev_action_last_chunk.shape}, cartesian_curr_action_first_chunk.shape: {cartesian_curr_action_first_chunk.shape}")
            # print(f"cartesian_prev_action_last_chunk[6]: {cartesian_prev_action_last_chunk[6]}, cartesian_curr_action_first_chunk[6]: {cartesian_curr_action_first_chunk[6]}")
            # print(f"cartesian_prev_action_last_chunk[13]: {cartesian_prev_action_last_chunk[13]}, cartesian_curr_action_first_chunk[13]: {cartesian_curr_action_first_chunk[13]}")
            left_ee_dist = np.linalg.norm(cartesian_prev_action_last_chunk[6] - cartesian_curr_action_first_chunk[6])
            right_ee_dist = np.linalg.norm(cartesian_prev_action_last_chunk[13] - cartesian_curr_action_first_chunk[13])
            print(f"Distance between {T} and {T + 1} chunks >> left end-effectors: {left_ee_dist:.4f}, right end-effectors: {right_ee_dist:.4f}")
            T += 1
        for action in actions:
            action = action * _AC_SCALE + _AC_LOC
            TASK_ENV.take_action(action)
            observation = TASK_ENV.get_obs()
            obs = encode_obs(observation, cam_indexes=cam_indexes)
            model.update_obs(obs)


def reset_model(model):  
    global act_history
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obs()
    act_history = None


_ROBOT_INSTANCE = None
_ARM_JOINT_NAMES = None

def get_cartesian_pose(joint_angles):
    """
    Calculates the Cartesian coordinates for the end-effectors (grippers).

    Args:
        joint_angles: numpy array of shape (14,)

    Returns:
        A tuple containing the Cartesian coordinates for the left and right end-effectors.
        (left_ee_pos, right_ee_pos), each is a numpy array of shape (3,).
    """
    global _ROBOT_INSTANCE, _ARM_JOINT_NAMES
    import yourdfpy

    if _ROBOT_INSTANCE is None:
        _ROBOT_INSTANCE = yourdfpy.URDF.load("/scratch2/meat124/dit_ws/src/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf")
        _ARM_JOINT_NAMES = sorted(
            [
                j
                for j in _ROBOT_INSTANCE.joint_names
                if ("fl_joint" in j or "fr_joint" in j)
                and any(char.isdigit() for char in j)
            ]
        )

    cfg = {}
    # Left arm (fl_joint1 to fl_joint6)
    for j in range(6):
        cfg[_ARM_JOINT_NAMES[j]] = joint_angles[j]
    # Left gripper (fl_joint7, fl_joint8) - controlled by a single angle
    cfg[_ARM_JOINT_NAMES[6]] = joint_angles[6]
    cfg[_ARM_JOINT_NAMES[7]] = joint_angles[6]

    # Right arm (fr_joint1 to fr_joint6)
    for j in range(6):
        cfg[_ARM_JOINT_NAMES[8 + j]] = joint_angles[7 + j]
    # Right gripper (fr_joint7, fr_joint8) - controlled by a single angle
    cfg[_ARM_JOINT_NAMES[14]] = joint_angles[13]
    cfg[_ARM_JOINT_NAMES[15]] = joint_angles[13]

    _ROBOT_INSTANCE.update_cfg(cfg)

    # Get positions for left gripper joints
    left_gripper_child_1 = _ROBOT_INSTANCE.joint_map[_ARM_JOINT_NAMES[6]].child
    left_gripper_child_2 = _ROBOT_INSTANCE.joint_map[_ARM_JOINT_NAMES[7]].child
    pos1 = _ROBOT_INSTANCE.get_transform(frame_to=left_gripper_child_1)[:3, 3]
    pos2 = _ROBOT_INSTANCE.get_transform(frame_to=left_gripper_child_2)[:3, 3]
    left_ee_pos = (pos1 + pos2) / 2.0

    # Get positions for right gripper joints
    right_gripper_child_1 = _ROBOT_INSTANCE.joint_map[_ARM_JOINT_NAMES[14]].child
    right_gripper_child_2 = _ROBOT_INSTANCE.joint_map[_ARM_JOINT_NAMES[15]].child
    pos1 = _ROBOT_INSTANCE.get_transform(frame_to=right_gripper_child_1)[:3, 3]
    pos2 = _ROBOT_INSTANCE.get_transform(frame_to=right_gripper_child_2)[:3, 3]
    right_ee_pos = (pos1 + pos2) / 2.0

    # The original function returned a (14, 3) array. To maintain compatibility
    # with the calling code that expects specific indices, we create a placeholder
    # array and populate only the required end-effector positions.
    joint_cartesian_coords = np.zeros((14, 3))
    joint_cartesian_coords[6] = left_ee_pos
    joint_cartesian_coords[13] = right_ee_pos

    return joint_cartesian_coords