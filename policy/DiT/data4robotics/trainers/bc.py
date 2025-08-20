# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from data4robotics.trainers.base import BaseTrainer


class BehaviorCloning(BaseTrainer):
    def training_step(self, batch, global_step):
        (imgs, obs), actions, mask, (ac_loc, ac_scale) = batch
        imgs = {k: v.to(self.device_id) for k, v in imgs.items()}
        obs, actions, mask = [ar.to(self.device_id) for ar in (obs, actions, mask)]
        ac_loc, ac_scale = ac_loc.to(self.device_id), ac_scale.to(self.device_id)
        ac_loc_expanded, ac_scale_expanded = ac_loc.unsqueeze(1), ac_scale.unsqueeze(1)

        # normalize actions
        obs = (obs - ac_loc) / ac_scale
        actions = (actions - ac_loc_expanded) / ac_scale_expanded

        ac_flat = actions.reshape((actions.shape[0], -1))
        mask_flat = mask.reshape((mask.shape[0], -1))
        loss = self.model(imgs, obs, ac_flat, mask_flat)
        self.log("bc_loss", global_step, loss.item())
        if self.is_train:
            self.log("lr", global_step, self.lr)
        return loss
