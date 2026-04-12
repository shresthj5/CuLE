import gc
import unittest

import torch
import torchcule_atari

from torchcule.atari import Env, Rom


class RomCompatibilityTests(unittest.TestCase):
    def test_legacy_and_modern_env_ids_resolve(self):
        expected_names = {
            "PongNoFrameskip-v4": "Video Olympics",
            "ALE/Pong-v5": "Video Olympics",
            "SpaceInvadersNoFrameskip-v4": "Space Invaders",
            "ALE/SpaceInvaders-v5": "Space Invaders",
            "MsPacmanNoFrameskip-v4": "Ms. Pac-Man",
            "ALE/MsPacman-v5": "Ms. Pac-Man",
            "BreakoutNoFrameskip-v4": "Breakout",
            "ALE/Breakout-v5": "Breakout",
        }

        for env_name, expected_fragment in expected_names.items():
            with self.subTest(env_name=env_name):
                rom = Rom(env_name)
                self.assertIn(expected_fragment, rom.game_name())
                self.assertTrue(rom.type_name())
                self.assertEqual(len(rom.md5()), 32)


class EnvSmokeTests(unittest.TestCase):
    def _assert_state_roundtrip(self, env_name, device_name):
        device = torch.device(device_name)
        env = Env(env_name, num_envs=2, color_mode="gray", device=device, rescale=False, frameskip=1)

        seeds = torch.tensor([123, 456], dtype=torch.int32, device=device)
        actions = torch.zeros(2, dtype=torch.uint8, device=device)

        env.reset(seeds=seeds, initial_steps=0)
        env.step(actions)
        states = torchcule_atari.AtariEnv.get_states(env, [0, 1])
        torchcule_atari.AtariEnv.set_states(env, [0, 1], states)
        obs, reward, done, info = env.step(actions)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        self.assertEqual(len(states), 2)
        self.assertEqual(tuple(obs.shape), (2, env.height, env.width, 1))
        self.assertEqual(tuple(reward.shape), (2,))
        self.assertEqual(tuple(done.shape), (2,))
        self.assertEqual(tuple(info["ale.lives"].shape), (2,))

    def test_extension_reports_supported_env_block_size(self):
        self.assertIn(int(torchcule_atari.ATARI_ENV_BLOCK_SIZE), {1, 32, 64, 128, 256})
        self.assertIn(int(torchcule_atari.ATARI_PROCESS_BLOCK_SIZE), {1, 32, 64, 128, 256})

    def test_v5_defaults_match_ale_semantics(self):
        env = Env("ALE/Pong-v5", num_envs=2, color_mode="gray", device="cpu", rescale=False)

        self.assertEqual(env.frameskip, 4)
        self.assertAlmostEqual(env.repeat_prob, 0.25)

    def test_sticky_action_wrapper_repeats_previous_action(self):
        env = Env("ALE/Pong-v5", num_envs=2, color_mode="gray", device="cpu", rescale=False, frameskip=4, repeat_prob=1.0)

        env.reset(seeds=torch.tensor([123, 456], dtype=torch.int32), initial_steps=0)
        env.last_actions.fill_(1)
        requested = torch.zeros(2, dtype=torch.uint8)
        sticky = env._apply_sticky_actions(requested, env.last_actions)

        self.assertEqual(sticky.tolist(), [1, 1])

    def test_cpu_reset_step_and_state_roundtrip(self):
        env = Env("ALE/Pong-v5", num_envs=2, color_mode="gray", device="cpu", rescale=False, frameskip=1)

        obs = env.reset(initial_steps=0)
        self.assertEqual(tuple(obs.shape), (2, 210, 160, 1))

        actions = torch.zeros(2, dtype=torch.uint8)
        obs2, reward, done, info = env.step(actions)

        self.assertEqual(tuple(obs2.shape), (2, 210, 160, 1))
        self.assertEqual(reward.tolist(), [0.0, 0.0])
        self.assertEqual(done.tolist(), [False, False])
        self.assertEqual(info["ale.lives"].tolist(), [0, 0])

        states = torchcule_atari.AtariEnv.get_states(env, [0, 1])
        torchcule_atari.AtariEnv.set_states(env, [0, 1], states)
        self.assertEqual(len(states), 2)

    def test_cpu_f8sc_state_roundtrip_uses_full_256_byte_ram(self):
        self._assert_state_roundtrip("ALE/ElevatorAction-v5", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this smoke test")
    def test_cuda_reset_step_and_state_roundtrip(self):
        env = Env("ALE/Pong-v5", num_envs=2, color_mode="gray", device="cuda:0", rescale=False, frameskip=1)

        seeds = torch.tensor([123, 456], dtype=torch.int32, device="cuda:0")
        env.reset(seeds=seeds, initial_steps=0)

        actions = torch.zeros(2, dtype=torch.uint8, device=env.device)
        obs, reward, done, info = env.step(actions)

        self.assertEqual(tuple(obs.shape), (2, 210, 160, 1))
        self.assertEqual(reward.tolist(), [0.0, 0.0])
        self.assertEqual(done.tolist(), [False, False])
        self.assertEqual(info["ale.lives"].tolist(), [0, 0])

        states = torchcule_atari.AtariEnv.get_states(env, [0, 1])
        torchcule_atari.AtariEnv.set_states(env, [0, 1], states)
        self.assertEqual(len(states), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this F8SC state bridge test")
    def test_cuda_f8sc_state_roundtrip_uses_full_256_byte_ram(self):
        self._assert_state_roundtrip("ALE/ElevatorAction-v5", "cuda:0")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this allocation test")
    def test_cuda_ram_buffer_pads_to_env_block_layout(self):
        env = Env("ALE/Pong-v5", num_envs=2, color_mode="gray", device="cuda:0", rescale=False, frameskip=1)
        block_size = int(torchcule_atari.ATARI_ENV_BLOCK_SIZE)
        padded_envs = 2 if block_size == 1 else block_size

        self.assertEqual(env.ram.numel(), padded_envs * env.cart.ram_size())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this migration test")
    def test_to_rejects_cross_device_env_migration_without_mutating_source_env(self):
        env = Env("ALE/Pong-v5", num_envs=2, color_mode="gray", device="cpu", rescale=False, frameskip=1)
        dense_ram_size = env.num_envs * env.cart.ram_size()

        env.reset(initial_steps=0)
        self.assertEqual(env.ram.numel(), dense_ram_size)

        with self.assertRaisesRegex(RuntimeError, "create the Env directly on the target device"):
            env.to("cuda:0")

        self.assertEqual(env.ram.device.type, "cpu")
        self.assertEqual(env.ram.numel(), dense_ram_size)

        actions = torch.zeros(env.num_envs, dtype=torch.uint8)
        obs, reward, done, info = env.step(actions)
        self.assertEqual(tuple(obs.shape), (2, 210, 160, 1))
        self.assertEqual(tuple(reward.shape), (2,))
        self.assertEqual(tuple(done.shape), (2,))
        self.assertEqual(tuple(info["ale.lives"].shape), (2,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this lifecycle test")
    def test_cuda_state_bridge_survives_repeated_env_lifecycles(self):
        for trial in range(3):
            env = Env("ALE/Pong-v5", num_envs=2, color_mode="gray", device="cuda:0", rescale=False, frameskip=1)
            seeds = torch.tensor([123 + trial, 456 + trial], dtype=torch.int32, device="cuda:0")
            env.reset(seeds=seeds, initial_steps=0)

            actions = torch.zeros(2, dtype=torch.uint8, device=env.device)
            env.step(actions)

            states = torchcule_atari.AtariEnv.get_states(env, [0, 1])
            torchcule_atari.AtariEnv.set_states(env, [0, 1], states)
            self.assertEqual(len(states), 2)

            del states
            del env
            gc.collect()


if __name__ == "__main__":
    unittest.main()
