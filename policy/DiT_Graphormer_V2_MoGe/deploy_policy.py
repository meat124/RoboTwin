# import packages and module here
import hydra
import torch
import sys, os
import yaml
import cv2
import numpy as np
import json

from .dit_model import DiT_Graphormer_V2_MoGe
import matplotlib.pyplot as plt
from collections import deque
from torchvision import transforms

from MoGe.moge.model.v2 import MoGeModel


IMAGE_SIZE = (256, 256)  # Fixed image size for resizing
EXP_WEIGHT = 0.01
_AC_LOC = None
_AC_SCALE = None
_DEPTH_MIN = None
_DEPTH_MAX = None
_NORMAL_MIN = None
_NORMAL_MAX = None
EPSILON = 1e-8


def get_preproc_transform(size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size), antialias=False),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


def encode_obs(observation, moge_model=None):  # Post-Process Observation
    def resize_obs(bgr_img: np.ndarray, size=IMAGE_SIZE):
        resized = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
        resized_rgb = resized[:, :, ::-1].copy()  # Convert BGR to RGB
        # resized = get_preproc_transform(size[0])(resized)
        # resized = resized.float() / 255.0  # Convert to CxHxW format and normalize
        return torch.from_numpy(resized_rgb).permute(2, 0, 1).float() / 255.0  # Convert to CxHxW format and normalize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_cam = resize_obs(observation["observation"]["head_camera"]["rgb"]) if "head_camera" in observation["observation"] else None
    obs = {
        "cam0": head_cam,
    }
    obs["agent_pos"] = torch.tensor(observation["joint_action"]["vector"], dtype=torch.float32)
    obs["agent_pos"] = (obs["agent_pos"] - _AC_LOC) / _AC_SCALE
    
    # infer depth, normal
    moge_model = moge_model.to(device)

    output = moge_model.infer(head_cam)
    depth = output.get("depth").cpu().numpy()
    normal = output.get("normal").cpu().numpy()
    mask = output.get("mask").cpu().numpy()

    # depth normalization
    normalized_depth = np.full_like(depth, 1.0, dtype=np.float32)  # background depth is 1.0
    valid_depth = depth[mask]
    normalized_value = (valid_depth - _DEPTH_MIN) / (_DEPTH_MAX - _DEPTH_MIN + EPSILON)
    normalized_depth[mask] = normalized_value
    normalized_depth = np.expand_dims(normalized_depth, axis=0).repeat(3, axis=0)  # 3xHxW for depth input

    # normal normalization
    normalized_normal = (normal - _NORMAL_MIN) / (_NORMAL_MAX - _NORMAL_MIN + EPSILON)
    normalized_normal = np.transpose(normalized_normal, (2, 0, 1))  # HxWx3 -> 3xHxW for normal input

    obs["cam1"] = torch.from_numpy(normalized_depth)
    obs["cam2"] = torch.from_numpy(normalized_normal)

    return obs


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    # temporally hardcode the path to the model checkpoint
    ckpt_file = usr_args['ckpt_setting']

    load_config_path = f"./policy/DiT_Graphormer_V2_MoGe/experiments/finetune.yaml"
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    ac_chunk = model_training_config['ac_chunk']
    img_chunk = model_training_config['img_chunk']

    import shutil
    norm_file = os.path.join(os.path.dirname(ckpt_file), "ac_norm.json")
    if os.path.exists(norm_file):
        shutil.copyfile(norm_file, "./ac_norm.json")

    model = DiT_Graphormer_V2_MoGe(
        ckpt_file=ckpt_file,
        ac_chunk=ac_chunk,
        img_chunk=img_chunk
    )

    return model


act_history = None


def eval(TASK_ENV, model, observation, temporal_ensemble=False, norm=None, moge_model=None):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    global act_history
    global _AC_LOC, _AC_SCALE, _DEPTH_MIN, _DEPTH_MAX, _NORMAL_MIN, _NORMAL_MAX
    _AC_LOC = norm["loc"]
    _AC_SCALE = norm["scale"]
    _DEPTH_MIN = norm["depth_min"]
    _DEPTH_MAX = norm["depth_max"]
    _NORMAL_MIN = norm["normal_min"]
    _NORMAL_MAX = norm["normal_max"]
    
    obs = encode_obs(observation, moge_model)
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
        obs = encode_obs(observation, moge_model)
        model.update_obs(obs)
    else:
        # ======== Get Action without Temporal Ensemble ========
        actions = model.get_action(obs)

        for action in actions:
            action = action * _AC_SCALE + _AC_LOC
            TASK_ENV.take_action(action)
            observation = TASK_ENV.get_obs()
            obs = encode_obs(observation, moge_model)
            model.update_obs(obs)


def reset_model(model):  
    global act_history
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obs()
    act_history = None
