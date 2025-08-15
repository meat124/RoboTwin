# import packages and module here
import hydra
import torch
import sys, os
import yaml
import cv2
import numpy as np

from .dit_model import DiT

IMAGE_SIZE = (256, 256)  # Fixed image size for resizing


def encode_obs(observation):  # Post-Process Observation
    def resize_obs(bgr_img: np.ndarray, size=IMAGE_SIZE):
        resized = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
        return torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).float() / 255.0

    front_cam = resize_obs(observation["observation"]["front_camera"]["rgb"]) if "front_camera" in observation["observation"] else None
    head_cam = resize_obs(observation["observation"]["head_camera"]["rgb"]) if "head_camera" in observation["observation"] else None
    left_cam = resize_obs(observation["observation"]["left_camera"]["rgb"]) if "left_camera" in observation["observation"] else None
    right_cam = resize_obs(observation["observation"]["right_camera"]["rgb"]) if "right_camera" in observation["observation"] else None
    obs = dict(
        cam0=front_cam,
        cam1=head_cam,
        cam2=left_cam,
        cam3=right_cam,
    )
    obs["agent_pos"] = torch.tensor(observation["joint_action"]["vector"], dtype=torch.float32)
    print(f"obs['agent_pos'].shape: {obs['agent_pos'].shape}")  # Debugging
    print(f"obs['agent_pos'].max(): {obs['agent_pos'].max()}, min: {obs['agent_pos'].min()}")  # Debugging
    print(f"obs['agent_pos'].mean(): {obs['agent_pos'].mean()}, std: {obs['agent_pos'].std()}")  # Debugging
    print(f"obs['agent_pos']: {obs['agent_pos']}")  # Debugging

    return obs


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    # temporally hardcode the path to the model checkpoint
    ckpt_file = usr_args['ckpt_setting']

    load_config_path = f"./policy/DiT/experiments/finetune.yaml"
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    ac_chunk = model_training_config['ac_chunk']
    img_chunk = model_training_config['img_chunk']

    model = DiT(
        ckpt_file=ckpt_file,
        ac_chunk=ac_chunk,
        img_chunk=img_chunk
    )

    return model


def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()

    actions = model.get_action(obs)  # Get Action according to observation chunk
    actions = actions[:5]

    for action in actions:  # Execute each step of the action
        # see for https://robotwin-platform.github.io/doc/control-robot.md more details
        TASK_ENV.take_action(action) # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        # TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obs()
