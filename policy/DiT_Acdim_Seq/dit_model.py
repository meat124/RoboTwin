import torch
import hydra
import yaml
import numpy as np
from pathlib import Path
from collections import deque


class DiT_Acdim_Seq:
    def __init__(self, ckpt_file: str, ac_chunk, img_chunk):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac_chunk = ac_chunk
        self.img_chunk = img_chunk

        self.obs_window = deque(maxlen=img_chunk + 1)

        self.policy = self.get_policy(ckpt_file)
        
        print(f"DiT_Acdim_Seq model loaded on device: {self.device}")

    def update_obs(self, observation):
        self.obs_window.append(observation)

    def reset_obs(self):
        self.obs_window.clear()

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def get_n_steps_obs(self):
        assert len(self.obs_window) > 0, "No observation is recorded, please update obs first"
        
        result = dict()
        for key in self.obs_window[0].keys():
            result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs_window], self.img_chunk)

        return result

    def get_action(self, obs):
        if obs is not None:
            self.obs_window.append(obs)  # update observation window
        obs = self.get_n_steps_obs()

        imgs = dict(
            cam0=obs["cam0"] if "cam0" in obs else None,
            cam1=obs["cam1"] if "cam1" in obs else None,
            cam2=obs["cam2"] if "cam2" in obs else None,
            cam3=obs["cam3"] if "cam3" in obs else None,
        )
        qpos = obs["agent_pos"][-1]

        imgs = {k: v.unsqueeze(0).to(self.device) for k, v in imgs.items() if v is not None}
        qpos = qpos.unsqueeze(0).to(self.device)  # add batch dimension
        
        with torch.no_grad():
            actions = self.policy.get_actions(imgs, qpos)

        return actions.detach().cpu().numpy().squeeze(0)

    def get_policy(self, ckpt_path: str):
        print(f"Loading policy from ckpt_path: {ckpt_path}")
        ckpt_path = Path(ckpt_path)    

        config_path = ckpt_path.parent / "agent_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find agent config file at {config_path}"
            )
        
        with open(config_path, "r") as f:
            agent_config = yaml.safe_load(f)

        model = hydra.utils.instantiate(agent_config)

        load_dict = torch.load(ckpt_path, map_location=self.device)

        if 'model' in load_dict:
            model.load_state_dict(load_dict['model'])
        else:
            raise ValueError("Could not find model state_dict in checkpoint")
        
        print(f"Successfully loaded model from {ckpt_path}. Global step: {load_dict.get('global_step', 'N/A')}")

        model.eval()
        model.to(self.device)
        
        return model


if __name__ == "__main__":
    # Example usage
    ckpt_file = "/scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT_Acdim_Seq/bc_finetune/dit_robotwin_stack_blocks_three_demo_clean_seed0/wandb_dit_robotwin_stack_blocks_three_demo_clean_seed0_stack_blocks_three_resnet_gn_2025-08-13_20-07-59/dit_robotwin_stack_blocks_three_demo_clean_seed0.ckpt"
    ac_chunk = 50  # Example value
    img_chunk = 3  # Example value
    model = DiT_Acdim_Seq(ckpt_file, ac_chunk, img_chunk)
    print("DiT_acdim model initialized.")
