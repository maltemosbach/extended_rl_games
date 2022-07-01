import numpy as np
import os
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import common_losses
from rl_games.algos_torch import torch_ext
from robomimic.utils.dataset import SequenceDataset
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import *


def dict_to_device(input_dict: Dict[str, Any],
                   device: torch.device) -> Dict[str, Any]:
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(device)
    return input_dict


class DAPGAgent(A2CAgent):
    def __init__(self, base_name: str, params: Dict[str, Any]) -> None:
        super().__init__(base_name, params)

        self.demo_cfg = params['config']['demo']
        self.demo_cfg['dataset']['goal_mode'] = None
        self.ppo_coef = params['config']['ppo_coef']
        self.bc_coef_const = params['config']['bc_coef_const']
        self.bc_coef_decay = params['config']['bc_coef_decay']
        self.l1_loss_weight = params['config']['l1_loss_weight']
        self.l2_loss_weight = params['config']['l2_loss_weight']
        self.load_demo_data()
        self.current_epoch = 0

    @property
    def bc_coef(self) -> float:
        return self.bc_coef_const * (self.bc_coef_decay ** self.current_epoch)

    def load_demo_data(self) -> None:
        train_dataset = SequenceDataset(**self.demo_cfg['dataset'],
                                        filter_by_attribute='train')
        valid_dataset = SequenceDataset(**self.demo_cfg['dataset'],
                                        filter_by_attribute='valid')

        train_sampler = train_dataset.get_dataset_sampler()
        valid_sampler = valid_dataset.get_dataset_sampler()

        self.demo_train_loader = DataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=self.demo_cfg['dataloader']['batch_size'],
            shuffle=(train_sampler is None),
            num_workers=self.demo_cfg['dataloader']['num_workers'],
            drop_last=True
        )
        self.demo_valid_loader = DataLoader(
            dataset=valid_dataset,
            sampler=valid_sampler,
            batch_size=self.demo_cfg['dataloader']['batch_size'],
            shuffle=(valid_sampler is None),
            num_workers=self.demo_cfg['dataloader']['num_workers'],
            drop_last=True
        )

        self.demo_train_iter = iter(self.demo_train_loader)
        self.demo_valid_iter = iter(self.demo_valid_loader)

    def sample_demo_dict(self, valid: bool = False) -> Dict[str, Any]:
        demo_iter = self.demo_valid_iter if valid else self.demo_train_iter
        try:
            demo_dict = next(demo_iter)
        except StopIteration:
            if valid:
                self.demo_valid_iter = iter(self.demo_valid_loader)
                demo_dict = next(self.demo_valid_iter)
            else:
                self.demo_train_iter = iter(self.demo_train_loader)
                demo_dict = next(self.demo_train_iter)
        return demo_dict

    def sample_demo_train_valid(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # demo train data
        demo_train_dict = self.sample_demo_dict()
        train_obs = self._preproc_obs(demo_train_dict['obs']['obs'])[:, 0]
        train_actions = demo_train_dict['actions'][:, 0]
        demo_train_dict = dict_to_device({
            'is_train': True,
            'prev_actions': train_actions,
            'obs': train_obs,
        }, device=self.ppo_device)

        # demo valid data
        demo_valid_dict = self.sample_demo_dict(valid=True)
        valid_obs = self._preproc_obs(demo_valid_dict['obs']['obs'])[:, 0]
        valid_actions = demo_valid_dict['actions'][:, 0]
        demo_valid_dict = dict_to_device({
            'is_train': False,
            'prev_actions': valid_actions,
            'obs': valid_obs,
        }, device=self.ppo_device)
        return demo_train_dict, demo_valid_dict

    def get_demo_actions(self, demo_train_dict: Dict[str, Any],
                         demo_valid_dict: Dict[str, Any] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        demo_train_res_dict = self.model(demo_train_dict)
        train_mu = demo_train_res_dict['mus']
        train_sigma = demo_train_res_dict['sigmas']

        if demo_valid_dict:
            with torch.no_grad():
                demo_valid_res_dict = self.model(demo_valid_dict)
                valid_mu = demo_valid_res_dict['mus']
                valid_sigma = demo_valid_res_dict['sigmas']
            return train_mu, valid_mu
        return train_mu

    def bc_loss(self, actions: torch.Tensor,
                target_actions: torch.Tensor) -> torch.Tensor:
        assert actions.shape == target_actions.shape
        l1_loss = nn.SmoothL1Loss()(actions, target_actions)
        l2_loss = nn.MSELoss()(actions, target_actions)
        bc_loss = self.l1_loss_weight * l1_loss + self.l2_loss_weight * l2_loss
        return bc_loss

    def calc_gradients(self, input_dict):
        # regular experience
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        # sample demonstration data
        demo_train_dict, demo_valid_dict = self.sample_demo_train_valid()

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # regular PPO losses
            res_dict = self.model(batch_dict)

            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch,
                                          action_log_probs, advantage, self.ppo,
                                          curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values,
                                                   curr_e_clip, return_batch,
                                                   self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1),
                 b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], \
                                              losses[3]

            ppo_loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            # Imitation loss
            demo_train_actions, demo_valid_actions = self.get_demo_actions(
                demo_train_dict, demo_valid_dict)

            bc_train_loss = self.bc_loss(demo_train_actions,
                                         demo_train_dict['prev_actions'])
            bc_valid_loss = self.bc_loss(demo_valid_actions,
                                         demo_valid_dict['prev_actions'])

            # Sum up bc and ppo loss
            loss = self.ppo_coef * ppo_loss + self.bc_coef * bc_train_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(),
                                          old_mu_batch, old_sigma_batch,
                                          reduce_kl)
            if rnn_masks is not None:
                kl_dist = (
                                      kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.diagnostics.mini_batch(self,
                                    {
                                        'values': value_preds_batch,
                                        'returns': return_batch,
                                        'new_neglogp': action_log_probs,
                                        'old_neglogp': old_action_log_probs_batch,
                                        'masks': rnn_masks
                                    }, curr_e_clip, 0)

        self.train_result = (a_loss, c_loss, entropy, \
                             kl_dist, self.last_lr, lr_mul, \
                             mu.detach(), sigma.detach(), b_loss,
                             bc_train_loss, bc_valid_loss)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            self.current_epoch = epoch_num
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, bc_train_losses, bc_valid_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                if self.print_stats:
                    step_time = max(step_time, 1e-6)
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.max_epochs}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                self.writer.add_scalar('losses/bc_train_loss', torch_ext.mean_list(bc_train_losses).item(), frame)
                self.writer.add_scalar('losses/bc_valid_loss', torch_ext.mean_list(bc_valid_losses).item(), frame)
                self.writer.add_scalar('info/bc_coef', self.bc_coef, frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        bc_train_losses = []
        bc_valid_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, bc_train_loss, bc_valid_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                bc_train_losses.append(bc_train_loss)
                bc_valid_losses.append(bc_valid_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.rank_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, bc_train_losses, bc_valid_losses, entropies, kls, last_lr, lr_mul