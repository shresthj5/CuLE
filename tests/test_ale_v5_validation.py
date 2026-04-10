import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
