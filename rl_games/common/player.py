import time
import gym
import numpy as np
import torch
import copy
from rl_games.common import env_configurations
from rl_games.algos_torch import  model_builder

class BasePlayer(object):
    def __init__(self, params):
        self.config = config = params['config']
        self.network_config = params['network']
        self.load_networks(params)
        self.env_name = self.config['env_name']
        self.env_config = self.config.get('env_config', {})
        self.env_info = self.config.get('env_info')
        self.clip_actions = config.get('clip_actions', True)
        self.seed = self.env_config.pop('seed', None)
        if self.env_info is None:
            self.env = self.create_env()
            self.env_info = env_configurations.get_env_info(self.env)
        else:
            self.env = config.get('vec_env')
        self.value_size = self.env_info.get('value_size', 1)
        self.action_space = self.env_info['action_space']
        self.num_agents = self.env_info['agents']

        self.observation_space = self.env_info['observation_space']
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get('central_value_config') is not None
        self.device_name = self.config.get('device_name', 'cuda')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 2000)
        self.is_determenistic = self.player_config.get('determenistic', True)
        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        self.max_steps = 108000 // 4
        self.device = torch.device(self.device_name)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k,v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_determenistic=False):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_determenistic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size(
            )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            #print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_determenistic)
                else:
                    action = self.get_action(obses, is_determenistic)

                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count,
                                  'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count,
                                  'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

    def teach(self, student_observation: str = 'privileged'):
        import torch.nn.functional as F
        from tqdm import tqdm
        from torch.utils.data import DataLoader
        from collections import defaultdict
        import wandb

        self.joint_pos_start_idx = self.infer_joint_angle_start_idx()

        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        obses = self.env.reset()
        batch_size = 1
        batch_size = self.get_batch_size(obses['obs'], batch_size)
        student = self.create_student(student_observation, obses)
        student_optimizer = torch.optim.Adam(student.parameters(), lr=0.00001)

        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

        cr = torch.zeros(batch_size, dtype=torch.float32, device=obses['obs'].device)
        steps = torch.zeros(batch_size, dtype=torch.float32, device=obses['obs'].device)

        dataset_dict = {}
        for epoch in range(1, 1001):
            #print("DAgger epoch:", epoch)
            save_every = 10
            if epoch % save_every == 0:
                torch.save(student.state_dict(), f"./student_dagger_epoch_{epoch}.pt")


            rollout_steps = 32
            obs_dict = defaultdict(list)
            expert_actions = []
            for step in range(rollout_steps):
                obs_dict = self.append_obs(obs_dict, obses, student_observation)

                act = self.mix_act(student, obs_dict, student_observation, p=1.0)
                expert_act = self.get_action(obses['obs'], self.is_determenistic)
                expert_actions.append(expert_act)

                obses, rew, done, info = self.env.step(act)
                cr += rew
                steps += 1

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices,
                                                        :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())

                    if self.print_stats:
                        print('reward:', cur_rewards / done_count,
                              'steps:', cur_steps / done_count)

            # After taking 32 rollout steps

            for k, v in obs_dict.items():
                obs_dict[k] = torch.cat(v).detach()
            expert_actions = torch.cat(expert_actions).detach()

            aggregate_dataset = True
            if aggregate_dataset:
                for k, v in obs_dict.items():
                    if k in dataset_dict.keys():
                        dataset_dict[k] = torch.cat([dataset_dict[k], v])
                    else:
                        dataset_dict[k] = v
                if "expert_actions" in dataset_dict.keys():
                    dataset_dict["expert_actions"] = torch.cat([dataset_dict["expert_actions"], expert_actions])
                else:
                    dataset_dict["expert_actions"] = expert_actions
            else:
                dataset_dict = obs_dict
                dataset_dict["expert_actions"] = expert_actions

            dataset_size = dataset_dict['obs'].shape[0]

            wandb.log({"dagger_dataset_size": dataset_dict['obs'].shape[0]})

            num_updates = 10000
            for update_step in range(num_updates):
                batch_i = torch.randint(0, dataset_size, size=(64,), device=dataset_dict['obs'].device)
                student_actions = self.get_student_action(student, dataset_dict, student_observation, batch_i)
                dagger_loss = F.mse_loss(student_actions, dataset_dict["expert_actions"][batch_i])
                dagger_loss.backward()
                student_optimizer.step()
                if update_step % 1000 == 0:
                    wandb.log({"dagger_loss": dagger_loss})
                    print("dagger_loss:", dagger_loss.item())

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if 'obs' in obses:
                obses = obses['obs']
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if 'observation' in obses:
                first_key = 'observation'
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size

    def get_num_channels(self, image_obses) -> int:
        _, num_cameras, _, _, num_channels = image_obses.shape
        return num_cameras * num_channels

    def infer_joint_angle_start_idx(self) -> int:
        # get object state size
        object_state_size = 0
        for obs in self.network_config['rn']['observations']:
            if obs.startswith('object'):
                object_state_size += self.network_config['rn']['obs_size'][obs]

        joint_angle_start_index = object_state_size * self.network_config['rn']['num_objects']

        for obs in self.network_config['rn']['observations']:
            if not obs.startswith('object'):
                if obs == 'jointPos':
                    break
                else:
                    joint_angle_start_index += self.network_config['rn']['obs_size'][obs]
        return joint_angle_start_index

    def mix_act(self, student, obs_dict, student_observation, p):
        if np.random.rand() < p:
            act = self.get_student_action(student, obs_dict, student_observation, batch_i=-1)
        else:
            act = self.get_action(obs_dict['obs'][-1], self.is_determenistic)
            print("getting expert action in mix_act")
        return act

    def create_student(self, student_observation: str, obses):
        from rl_games.algos_torch.students import Impala2D, MLPStudent
        action_size = 11

        if student_observation == 'privileged':
            state_size = obses['obs'].shape[1]
            student = MLPStudent(state_size, action_size).to(
                obses['obs'].device).train()
        elif student_observation == 'image_and_joint_pos':
            num_channels = self.get_num_channels(obses['image'])
            image_size = obses['image'].shape[2]
            state_size = 17
            student = Impala2D(
                num_channels, 32, image_size, state_size, action_size).to(
                obses['obs'].device).train()
        elif student_observation == 'voxelgrid':
            raise NotImplementedError
        else:
            assert False
        return student

    def append_obs(self, obs_dict, obses, student_observation):
        obs_dict['obs'].append(obses['obs'])

        if student_observation == 'images_and_joint_pos':
            images = torch.cat([obses['image'][:, cam] for cam in
                                range(obses['image'].shape[1])],
                               dim=-1).permute(0, 3, 1, 2)
            joint_pos = obses['obs'][:,
                        self.joint_pos_start_idx:self.joint_pos_start_idx + 17]
            obs_dict['images'].append(images)
            obs_dict['joint_pos'].append(joint_pos)
        return obs_dict

    def get_student_action(self, student, obs_dict, student_observation, batch_i = None):
        if batch_i is None:
            batch_i = torch.arange(0, obs_dict['obs'].shape[0], device=obs_dict['obs'].device)
        if student_observation == 'privileged':
            act = student(obs_dict['obs'][batch_i])
        elif student_observation == 'image_and_joint_pos':
            act = student(obs_dict['image'][batch_i],
                          obs_dict['joint_pos'][batch_i])
        else:
            assert False
        return torch.clamp(act, -1, 1)
