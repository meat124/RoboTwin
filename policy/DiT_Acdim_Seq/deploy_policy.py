# import packages and module here
import hydra
import torch
import sys, os
import yaml
import cv2
import numpy as np
import json

from .dit_model import DiT_Acdim_Seq
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

    load_config_path = f"./policy/DiT_Acdim_Seq/experiments/finetune.yaml"
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    ac_chunk = model_training_config['ac_chunk']
    img_chunk = model_training_config['img_chunk']

    import shutil
    norm_file = os.path.join(os.path.dirname(ckpt_file), "ac_norm.json")
    if os.path.exists(norm_file):
        shutil.copyfile(norm_file, "./ac_norm.json")

    model = DiT_Acdim_Seq(
        ckpt_file=ckpt_file,
        ac_chunk=ac_chunk,
        img_chunk=img_chunk
    )

    return model


act_history = None


def eval(TASK_ENV, model, observation, temporal_ensemble=False, norm=None, cam_indexes=[0, 1, 2, 3]):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    global act_history
    global _AC_LOC, _AC_SCALE
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
