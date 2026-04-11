import unittest

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torchcule.atari import Env

from tools.validate_ale_v5_subset import (
    validate_metadata,
    validate_restored_cpu_cuda,
    validate_state_roundtrip,
)


class AleV5ValidationTests(unittest.TestCase):
    def test_supported_v5_metadata_matches_ale_py(self):
        self.assertEqual(validate_metadata(), [])

    def test_supported_v5_public_state_roundtrip_is_idempotent(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda:0")

        self.assertEqual(validate_state_roundtrip(devices), [])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CPU/CUDA parity validation")
    def test_restored_cpu_cuda_rewards_done_and_lives_match(self):
        result = validate_restored_cpu_cuda(steps=8, num_envs=2)
        self.assertEqual(result["transition_failures"], [])

    def test_qbert_early_frames_match_ale(self):
        gym.register_envs(ale_py)

        ale_env = gym.make(
            "ALE/Qbert-v5",
            obs_type="grayscale",
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
        )
        cule_env = Env(
            "ALE/Qbert-v5",
            1,
            color_mode="gray",
            device="cpu",
            rescale=False,
            frameskip=1,
            repeat_prob=0.0,
        )

        ale_obs, _ = ale_env.reset(seed=123)
        cule_obs = cule_env.reset(
            seeds=torch.tensor([123], dtype=torch.int32),
            initial_steps=0,
        ).cpu().numpy()[0, :, :, 0]
        np.testing.assert_array_equal(ale_obs, cule_obs)

        for _ in range(5):
            ale_obs, _, _, _, _ = ale_env.step(0)
            cule_obs, _, _, _ = cule_env.step(torch.tensor([0], dtype=torch.uint8))
            np.testing.assert_array_equal(ale_obs, cule_obs.cpu().numpy()[0, :, :, 0])


if __name__ == "__main__":
    unittest.main()
