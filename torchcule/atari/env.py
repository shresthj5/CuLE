"""CuLE (CUda Learning Environment module).

This module provides access to several RL environments that generate data
on the CPU or GPU.
"""

import math
import re
import numpy as np
import torch

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover - compatibility fallback
    from gym import spaces  # type: ignore[no-redef]

from torchcule.atari.rom import Rom
import torchcule_atari


_UINT32_MODULUS = 1 << 32
_ALE_V5_SYSTEM_RANDOM_SEED = 4753849


def _is_ale_v5_env(env_name):
    return re.match(r"^ALE\/[A-Za-z0-9]+-v5$", env_name) is not None


def _default_semantics(env_name, frameskip, repeat_prob):
    if _is_ale_v5_env(env_name):
        if frameskip is None:
            frameskip = 4
        if repeat_prob is None:
            repeat_prob = 0.25
    else:
        if frameskip is None:
            frameskip = 1
        if repeat_prob is None:
            repeat_prob = 0.0

    return frameskip, repeat_prob


def _map_ale_v5_seed(seed_value):
    state = np.random.SeedSequence(int(seed_value)).generate_state(2)
    return np.asarray(state, dtype=np.int32)[1].item()


class Env(torchcule_atari.AtariEnv):
    """
    ALE (Atari Learning Environment)

    This class provides access to ALE environments that may be executed on the CPU
    or GPU.

    Example:
		import argparse
		from torchcule.atari import Env

		parser = argparse.ArgumentParser(description="CuLE")
		parser.add_argument("game", type=str,
							help="Atari game name (breakout)")
		parser.add_argument("--n", type=int, default=20,
							help="Number of atari environments")
		parser.add_argument("--s", type=int, default=200,
							help="Number steps/frames to generate per environment")
		parser.add_argument("--c", type=str, default='rgb',
							help="Color mode (rgb or gray)")
		parser.add_argument("--rescale", action='store_true',
							help="Resize output frames to 84x84 using bilinear interpolation")
		args = parser.parse_args()

		color_mode = args.c
		num_envs = args.n
		num_steps = args.s

		env = Env(args.game, num_envs, color_mode, args.rescale)
		observations = env.reset()

		for _ in np.arange(num_steps):
			actions = env.sample_random_actions()
			observations, reward, done, info = env.step(actions)
    """

    def __init__(self, env_name, num_envs, color_mode='rgb', device='cpu', rescale=False,
                 frameskip=None, repeat_prob=None, episodic_life=False, max_noop_steps=30,
                 max_episode_length=10000):
        """Initialize the ALE class with a given environment

        Args:
            env_name (str): The name of the Atari rom
            num_envs (int): The number of environments to run
            color_mode (str): RGB ('rgb') or grayscale ('gray') observations
            use_cuda (bool) : Map ALEs to GPU
            rescale (bool) : Rescale grayscale observations to 84x84
            frameskip (int) : Number of frames to skip during training
            repeat_prob (float) : Probability of repeating previous action
            clip_rewards (bool) : Apply rewards clipping to {-1,1}
            episodic_life (bool) : Set 'done' on end of life
        """

        assert (color_mode == 'rgb') or (color_mode == 'gray')
        if color_mode == 'rgb' and rescale:
            raise ValueError('Rescaling is only valid in grayscale color mode')

        frameskip, repeat_prob = _default_semantics(env_name, frameskip, repeat_prob)
        if _is_ale_v5_env(env_name) and max_noop_steps == 30:
            max_noop_steps = 1

        self.cart = Rom(env_name)
        self.is_ale_v5_env = _is_ale_v5_env(env_name)
        super(Env, self).__init__(self.cart, num_envs, max_noop_steps)

        self.device = torch.device(device)
        self.num_envs = num_envs
        self.rescale = rescale
        self.frameskip = frameskip
        self.repeat_prob = repeat_prob
        self.is_cuda = self.device.type == 'cuda'
        self.is_training = False
        self.episodic_life = episodic_life
        self.height = 84 if self.rescale else self.cart.screen_height()
        self.width = 84 if self.rescale else self.cart.screen_width()
        self.num_channels = 3 if color_mode == 'rgb' else 1

        self.action_set = torch.Tensor([int(s) for s in self.cart.minimal_actions()]).to(self.device).byte()
        noop_matches = torch.nonzero(self.action_set == int(torchcule_atari.NOOP), as_tuple=False)
        if noop_matches.numel() == 0:
            raise ValueError("The minimal action set must contain NOOP for ALE-compatible semantics")
        self.noop_action_index = int(noop_matches[0].item())

        # check if FIRE is in the action set
        self.fire_reset = int(torchcule_atari.FIRE) in self.action_set

        self.action_space = spaces.Discrete(self.action_set.size(0))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.num_channels, self.height, self.width), dtype=np.uint8)

        self.observations1 = torch.zeros((num_envs, self.height, self.width, self.num_channels), device=self.device, dtype=torch.uint8)
        self.observations2 = torch.zeros((num_envs, self.height, self.width, self.num_channels), device=self.device, dtype=torch.uint8)
        self.done = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.actions = torch.zeros(num_envs, device=self.device, dtype=torch.uint8)
        self.last_actions = torch.full((num_envs,), self.noop_action_index, device=self.device, dtype=torch.uint8)
        self.last_player_b_actions = torch.full((num_envs,), self.noop_action_index, device=self.device, dtype=torch.uint8)
        self.noop_action_indices = torch.full((num_envs,), self.noop_action_index, device=self.device, dtype=torch.uint8)
        self.player_b_action_indices = torch.full((num_envs,), self.noop_action_index, device=self.device, dtype=torch.uint8)
        self.player_a_actions = torch.zeros(num_envs, device=self.device, dtype=torch.uint8)
        self.player_b_actions = torch.zeros(num_envs, device=self.device, dtype=torch.uint8)
        self.lives = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self.rewards = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self._sticky_random_states = None
        self._sticky_rng_seeded = False
        self._sticky_threshold = int(float(self.repeat_prob) * _UINT32_MODULUS)

        self.states = torch.zeros((num_envs, self.state_size()), device=self.device, dtype=torch.uint8)
        self.frame_states = torch.zeros((num_envs, self.frame_state_size()), device=self.device, dtype=torch.uint8)
        self.ram = torch.randint(0, 255, (self._ram_buffer_size(self.device),), device=self.device, dtype=torch.uint8)
        self.tia = torch.zeros((num_envs, self.tia_update_size()), device=self.device, dtype=torch.int32)
        self.frame_buffer = torch.zeros((num_envs, 300 * self.cart.screen_width()), device=self.device, dtype=torch.uint8)
        self._previous_frame_buffer = torch.zeros((num_envs, 300 * self.cart.screen_width()), device=self.device, dtype=torch.uint8)
        self.cart_offsets = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self.rand_states = torch.randint(0, np.iinfo(np.int32).max, (num_envs,), device=self.device, dtype=torch.int32)
        self.cached_states = torch.zeros((max_noop_steps, self.state_size()), device=self.device, dtype=torch.uint8)
        self.cached_ram = torch.randint(0, 255, (max_noop_steps, self.cart.ram_size()), device=self.device, dtype=torch.uint8)
        self.cached_frame_states = torch.zeros((max_noop_steps, self.frame_state_size()), device=self.device, dtype=torch.uint8)
        self.cached_frames = torch.zeros((max_noop_steps, 300 * self.cart.screen_width()), device=self.device, dtype=torch.uint8)
        self._cached_previous_frames = torch.zeros((max_noop_steps, 300 * self.cart.screen_width()), device=self.device, dtype=torch.uint8)
        self.cached_tia = torch.zeros((max_noop_steps, self.tia_update_size()), device=self.device, dtype=torch.int32)
        self.cache_index = torch.zeros((num_envs,), device=self.device, dtype=torch.int32)

        self.set_cuda(self.is_cuda, self._cuda_device_index(self.device))
        self.initialize(self.states.data_ptr(),
                        self.frame_states.data_ptr(),
                        self.ram.data_ptr(),
                        self.tia.data_ptr(),
                        self.frame_buffer.data_ptr(),
                        self._previous_frame_buffer.data_ptr(),
                        self.cart_offsets.data_ptr(),
                        self.action_set.data_ptr(),
                        self.rand_states.data_ptr(),
                        self.cached_states.data_ptr(),
                        self.cached_ram.data_ptr(),
                        self.cached_frame_states.data_ptr(),
                        self.cached_frames.data_ptr(),
                        self._cached_previous_frames.data_ptr(),
                        self.cached_tia.data_ptr(),
                        self.cache_index.data_ptr());

    def _ram_buffer_size(self, device):
        device = torch.device(device)
        num_ram_envs = self._ram_env_capacity(device)

        return num_ram_envs * self.cart.ram_size()

    def _cuda_device_index(self, device):
        device = torch.device(device)

        if device.type != 'cuda':
            return -1

        return torch.cuda.current_device() if device.index is None else device.index

    def _ram_env_capacity(self, device):
        device = torch.device(device)
        env_block_size = int(torchcule_atari.ATARI_ENV_BLOCK_SIZE)

        if device.type == 'cuda' and env_block_size > 1:
            return math.ceil(self.num_envs / env_block_size) * env_block_size

        return self.num_envs

    def _same_device(self, other_device):
        other_device = torch.device(other_device)

        if self.device.type != other_device.type:
            return False

        if self.device.type == 'cuda':
            return self._cuda_device_index(self.device) == self._cuda_device_index(other_device)

        return self.device.index == other_device.index

    def to(self, device):
        device = torch.device(device)

        if not self._same_device(device) and (self.device.type == 'cuda' or device.type == 'cuda'):
            raise RuntimeError(
                'Moving an existing Env between CPU/CUDA devices is not supported because '
                'CuLE state caches contain device-specific runtime pointers; create the Env '
                'directly on the target device instead.'
            )

        if self.is_cuda:
            torch.cuda.current_stream().synchronize()
            self.sync_this_stream()
            self.sync_other_stream()

        self.device = device
        self.is_cuda = self.device.type == 'cuda'
        self.set_cuda(self.is_cuda, self._cuda_device_index(self.device))

        self.observations1 = self.observations1.to(self.device)
        self.observations2 = self.observations2.to(self.device)
        self.done = self.done.to(self.device)
        self.actions = self.actions.to(self.device)
        self.last_actions = self.last_actions.to(self.device)
        self.last_player_b_actions = self.last_player_b_actions.to(self.device)
        self.noop_action_indices = self.noop_action_indices.to(self.device)
        self.player_b_action_indices = self.player_b_action_indices.to(self.device)
        self.player_a_actions = self.player_a_actions.to(self.device)
        self.player_b_actions = self.player_b_actions.to(self.device)
        self.lives = self.lives.to(self.device)
        self.rewards = self.rewards.to(self.device)
        self.action_set = self.action_set.to(self.device)

        self.states = self.states.to(self.device)
        self.frame_states = self.frame_states.to(self.device)
        self.ram = self.ram.to(self.device)
        self.tia = self.tia.to(self.device)
        self.frame_buffer = self.frame_buffer.to(self.device)
        self._previous_frame_buffer = self._previous_frame_buffer.to(self.device)
        self.cart_offsets = self.cart_offsets.to(self.device)
        self.rand_states = self.rand_states.to(self.device)
        self.cached_states = self.cached_states.to(self.device)
        self.cached_ram = self.cached_ram.to(self.device)
        self.cached_frame_states = self.cached_frame_states.to(self.device)
        self.cached_frames = self.cached_frames.to(self.device)
        self._cached_previous_frames = self._cached_previous_frames.to(self.device)
        self.cached_tia = self.cached_tia.to(self.device)
        self.cache_index = self.cache_index.to(self.device)

        self.initialize(self.states.data_ptr(),
                        self.frame_states.data_ptr(),
                        self.ram.data_ptr(),
                        self.tia.data_ptr(),
                        self.frame_buffer.data_ptr(),
                        self._previous_frame_buffer.data_ptr(),
                        self.cart_offsets.data_ptr(),
                        self.action_set.data_ptr(),
                        self.rand_states.data_ptr(),
                        self.cached_states.data_ptr(),
                        self.cached_ram.data_ptr(),
                        self.cached_frame_states.data_ptr(),
                        self.cached_frames.data_ptr(),
                        self._cached_previous_frames.data_ptr(),
                        self.cached_tia.data_ptr(),
                        self.cache_index.data_ptr());

        if self.is_cuda:
            torch.cuda.current_stream().synchronize()
            self.sync_this_stream()
            self.sync_other_stream()

    def train(self, frameskip=4):
        """Set ALE to training mode"""
        self.frameskip = frameskip
        self.is_training = True

    def eval(self):
        """Set ALE to evaluation mode"""
        self.is_training = False

    def minimal_actions(self):
        """Minimal number of actions for the environment

        Returns:
            list[Action]: minimal set of actions for the environment
        """
        return self.action_set

    def sample_random_actions(self, asyn=False):
        """Generate a random set of actions

        Returns:
            list[Action]: random set of actions generated for the environment
        """
        return torch.randint(self.minimal_actions().size(0), (self.num_envs,), device=self.device, dtype=torch.uint8)

    def screen_shape(self):
        """Get the shape of the observations

        Returns:
            tuple(int,int): Tuple containing height and width of observations
        """
        return (self.height, self.width)

    def reset(self, seeds=None, initial_steps=50, verbose=False, asyn=False):
        """Reset the environments

        Args:
            seeds (list[int]): seeds to use for initialization
            initial_steps (int): number of initial NOOP steps to execute during initialization

        Returns:
            tuple(int,int): Tuple containing height and width of observations
        """
        if seeds is None:
            seeds = torch.randint(np.iinfo(np.int32).max, (self.num_envs,), dtype=torch.int32, device=self.device)
        elif not isinstance(seeds, torch.Tensor):
            seeds = torch.as_tensor(seeds, dtype=torch.int32, device=self.device)

        host_requested_seeds = seeds.detach().cpu().tolist()
        if self.is_ale_v5_env:
            host_ale_seeds = [_map_ale_v5_seed(seed) for seed in host_requested_seeds]
            # ALE v5 reseeds the environment RNG on reset but keeps Stella's
            # emulator-core system_random_seed fixed for deterministic reset state.
            reset_seeds = torch.full(
                (self.num_envs,),
                _ALE_V5_SYSTEM_RANDOM_SEED,
                dtype=torch.int32,
                device=self.device,
            )
            ale_reset_seeds = torch.as_tensor(
                host_ale_seeds,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            host_ale_seeds = host_requested_seeds
            reset_seeds = seeds
            ale_reset_seeds = seeds

        if self.repeat_prob > 0.0:
            self.seed_sticky_actions(
                ale_reset_seeds.data_ptr(),
                self.is_ale_v5_env,
                int(self._sticky_threshold),
            )
            self._sticky_rng_seeded = self.is_ale_v5_env
        else:
            self.seed_sticky_actions(0, False, 0)
            self._sticky_rng_seeded = False

        if self.is_cuda:
            self.sync_other_stream()
            stream = torch.cuda.current_stream()

        self.configure_reset_semantics(
            self.is_ale_v5_env,
            int(self.frameskip),
            float(self.repeat_prob),
        )
        super(Env, self).reset(reset_seeds.data_ptr(), ale_reset_seeds.data_ptr())
        self.last_actions.fill_(self.noop_action_index)
        self.last_player_b_actions.fill_(self.noop_action_index)
        self.observations1.zero_()
        self.observations2.zero_()
        if self.is_cuda:
            self.sync_other_stream()
        self.generate_frames(self.rescale, True, self.num_channels, self.observations1.data_ptr())

        if self.is_training:
            iterator = range(math.ceil(initial_steps / self.frameskip))

            if verbose:
                from tqdm import tqdm
                iterator = tqdm(iterator)

            for _ in iterator:
                actions = self.sample_random_actions()
                self.step(actions, asyn=True)

        if self.is_cuda:
            self.sync_this_stream()
            if not asyn:
                stream.synchronize()

        return self.observations1

    def _apply_sticky_actions(self, requested_actions, previous_actions, output=None):
        requested_actions = requested_actions.to(self.device, dtype=torch.uint8)
        if output is None:
            output = torch.empty_like(requested_actions, device=self.device)

        if self.repeat_prob <= 0.0:
            output.copy_(requested_actions)
            return output

        if self.is_ale_v5_env and self._sticky_rng_seeded:
            self.apply_exact_sticky_actions(
                requested_actions.data_ptr(),
                previous_actions.data_ptr(),
                output.data_ptr(),
            )
        else:
            sticky_mask = torch.rand(self.num_envs, device=self.device) < self.repeat_prob
            output.copy_(torch.where(sticky_mask, previous_actions, requested_actions))

        return output

    def step(self, player_a_actions, player_b_actions=None, asyn=False):
        """Take a step in the environment by apply a set of actions

        Args:
            actions (list[Action]): list of actions to apply to each environment

        Returns:
            ByteTensor: observations for each environment
            IntTensor: sum of rewards for frameskip steps in each environment
            ByteTensor: 'done' state for each environment
            list[str]: miscellaneous information (currently unused)
        """

	    # sanity checks
        assert player_a_actions.size(0) == self.num_envs

        self.rewards.zero_()
        self.observations1.zero_()
        self.observations2.zero_()
        self.done.zero_()

        for frame in range(self.frameskip):
            self._apply_sticky_actions(player_a_actions, self.last_actions, output=self.actions)
            torch.index_select(self.action_set, 0, self.actions.long(), out=self.player_a_actions)
            player_a_actions_ptr = self.player_a_actions.data_ptr()
            self.last_actions.copy_(self.actions)

            if player_b_actions is not None:
                requested_player_b_actions = player_b_actions
            else:
                requested_player_b_actions = self.noop_action_indices

            self._apply_sticky_actions(
                requested_player_b_actions,
                self.last_player_b_actions,
                output=self.player_b_action_indices,
            )
            self.last_player_b_actions.copy_(self.player_b_action_indices)

            if player_b_actions is not None:
                torch.index_select(self.action_set, 0, self.player_b_action_indices.long(), out=self.player_b_actions)
                player_b_actions_ptr = self.player_b_actions.data_ptr()
            else:
                player_b_actions_ptr = 0

            if self.is_cuda:
                self.sync_other_stream()
            super(Env, self).step(self.fire_reset and self.is_training, player_a_actions_ptr, player_b_actions_ptr, self.done.data_ptr())
            self.get_data(self.episodic_life, self.done.data_ptr(), self.rewards.data_ptr(), self.lives.data_ptr())
            if self.is_ale_v5_env:
                if frame != (self.frameskip - 1):
                    # ALE v5 returns the final repeated frame without max-pooling,
                    # but the replay state still has to advance across intermediate
                    # subframes so the last-frame render starts from the correct
                    # framebuffer/TIA state.
                    self.generate_frames(self.rescale, False, self.num_channels, self.observations2.data_ptr())
            elif frame == (self.frameskip - 2):
                self.generate_frames(self.rescale, False, self.num_channels, self.observations2.data_ptr())

        self.reset_states()
        self.generate_frames(self.rescale, True, self.num_channels, self.observations1.data_ptr())

        if self.is_cuda:
            self.sync_this_stream()
            if not asyn:
                torch.cuda.current_stream().synchronize()

        if not self.is_ale_v5_env:
            self.observations1 = torch.max(self.observations1, self.observations2)

        info = {'ale.lives': self.lives}

        return self.observations1, self.rewards, self.done, info

    def get_states(self, indices):
        from torchcule.atari.state import State
        return [State(s) for s in super(Env, self).get_states([i for i in indices.cpu()])]

    def set_states(self, indices, states):
        super(Env, self).set_states([i for i in indices.cpu()], [s.state for s in states])
