import unittest

import torch

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility fallback
    import gym  # type: ignore[no-redef]

import ale_py

from torchcule.atari import Env, Rom


SUPPORTED_V5_ENVS = (
    "ALE/Adventure-v5",
    "ALE/AirRaid-v5",
    "ALE/Alien-v5",
    "ALE/Amidar-v5",
    "ALE/Assault-v5",
    "ALE/Asterix-v5",
    "ALE/Asteroids-v5",
    "ALE/Atlantis-v5",
    "ALE/BankHeist-v5",
    "ALE/BattleZone-v5",
    "ALE/BeamRider-v5",
    "ALE/Berzerk-v5",
    "ALE/Bowling-v5",
    "ALE/Boxing-v5",
    "ALE/Breakout-v5",
    "ALE/Carnival-v5",
    "ALE/Centipede-v5",
    "ALE/ChopperCommand-v5",
    "ALE/CrazyClimber-v5",
    "ALE/Defender-v5",
    "ALE/DemonAttack-v5",
    "ALE/DoubleDunk-v5",
    "ALE/ElevatorAction-v5",
    "ALE/Enduro-v5",
    "ALE/FishingDerby-v5",
    "ALE/Freeway-v5",
    "ALE/Frostbite-v5",
    "ALE/Gopher-v5",
    "ALE/Gravitar-v5",
    "ALE/Hero-v5",
    "ALE/IceHockey-v5",
    "ALE/Jamesbond-v5",
    "ALE/JourneyEscape-v5",
    "ALE/Kaboom-v5",
    "ALE/Kangaroo-v5",
    "ALE/Krull-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MontezumaRevenge-v5",
    "ALE/MsPacman-v5",
    "ALE/NameThisGame-v5",
    "ALE/Phoenix-v5",
    "ALE/Pitfall-v5",
    "ALE/Pong-v5",
    "ALE/Pooyan-v5",
    "ALE/PrivateEye-v5",
    "ALE/Qbert-v5",
    "ALE/Riverraid-v5",
    "ALE/RoadRunner-v5",
    "ALE/Robotank-v5",
    "ALE/Seaquest-v5",
    "ALE/Skiing-v5",
    "ALE/Solaris-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/StarGunner-v5",
    "ALE/Tennis-v5",
    "ALE/TimePilot-v5",
    "ALE/Tutankham-v5",
    "ALE/UpNDown-v5",
    "ALE/Venture-v5",
    "ALE/VideoPinball-v5",
    "ALE/WizardOfWor-v5",
    "ALE/YarsRevenge-v5",
    "ALE/Zaxxon-v5",
)

UNSUPPORTED_V5_ENVS = (
    "ALE/Atlantis2-v5",
    "ALE/Backgammon-v5",
    "ALE/BasicMath-v5",
    "ALE/Blackjack-v5",
    "ALE/Casino-v5",
    "ALE/Crossbow-v5",
    "ALE/Darkchambers-v5",
    "ALE/DonkeyKong-v5",
    "ALE/Earthworld-v5",
    "ALE/Entombed-v5",
    "ALE/Et-v5",
    "ALE/FlagCapture-v5",
    "ALE/Frogger-v5",
    "ALE/Galaxian-v5",
    "ALE/Hangman-v5",
    "ALE/HauntedHouse-v5",
    "ALE/HumanCannonball-v5",
    "ALE/KeystoneKapers-v5",
    "ALE/KingKong-v5",
    "ALE/Klax-v5",
    "ALE/Koolaid-v5",
    "ALE/LaserGates-v5",
    "ALE/LostLuggage-v5",
    "ALE/MarioBros-v5",
    "ALE/MiniatureGolf-v5",
    "ALE/MrDo-v5",
    "ALE/Othello-v5",
    "ALE/Pacman-v5",
    "ALE/Pitfall2-v5",
    "ALE/SirLancelot-v5",
    "ALE/SpaceWar-v5",
    "ALE/Superman-v5",
    "ALE/Surround-v5",
    "ALE/Tetris-v5",
    "ALE/TicTacToe3D-v5",
    "ALE/Trondead-v5",
    "ALE/Turmoil-v5",
    "ALE/VideoCheckers-v5",
    "ALE/VideoChess-v5",
    "ALE/VideoCube-v5",
    "ALE/WordZapper-v5",
)


def _registered_v5_envs():
    gym.register_envs(ale_py)
    return sorted(
        spec.id
        for spec in gym.envs.registry.values()
        if spec.id.startswith("ALE/") and spec.id.endswith("-v5")
    )


def _exercise_env(env_name, device_name):
    device = torch.device(device_name)
    env = Env(
        env_name,
        num_envs=2,
        color_mode="gray",
        device=device,
        rescale=False,
        frameskip=1,
        repeat_prob=0.0,
    )

    seeds = torch.tensor([123, 456], dtype=torch.int32, device=device)
    actions = torch.zeros(2, dtype=torch.uint8, device=device)

    obs = env.reset(seeds=seeds, initial_steps=0)
    next_obs, reward, done, info = env.step(actions)

    expected_shape = (2, env.height, env.width, 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return env, obs, next_obs, reward, done, info, expected_shape


class AleV5CoverageTests(unittest.TestCase):
    def test_registered_v5_envs_match_audited_support_partition(self):
        self.assertEqual(
            _registered_v5_envs(),
            sorted(SUPPORTED_V5_ENVS + UNSUPPORTED_V5_ENVS),
        )

    def test_supported_v5_roms_resolve(self):
        for env_name in SUPPORTED_V5_ENVS:
            with self.subTest(env_name=env_name):
                rom = Rom(env_name)
                self.assertEqual(len(rom.md5()), 32)
                self.assertTrue(rom.game_name())
                self.assertTrue(rom.is_supported())
                self.assertNotEqual(rom.type_name(), "NOT_SUPPORTED")

    def test_unsupported_v5_roms_raise_clear_error(self):
        for env_name in UNSUPPORTED_V5_ENVS:
            with self.subTest(env_name=env_name):
                with self.assertRaisesRegex(ValueError, r"CuLE does not currently support"):
                    Rom(env_name)

    def test_all_supported_v5_envs_cpu_reset_and_step(self):
        for env_name in SUPPORTED_V5_ENVS:
            with self.subTest(env_name=env_name):
                env, obs, next_obs, reward, done, info, expected_shape = _exercise_env(env_name, "cpu")
                self.assertEqual(tuple(obs.shape), expected_shape)
                self.assertEqual(tuple(next_obs.shape), expected_shape)
                self.assertEqual(tuple(reward.shape), (env.num_envs,))
                self.assertEqual(tuple(done.shape), (env.num_envs,))
                self.assertEqual(tuple(info["ale.lives"].shape), (env.num_envs,))
                self.assertTrue(bool(torch.isfinite(reward).all().item()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this v5 coverage test")
    def test_all_supported_v5_envs_cuda_reset_and_step(self):
        for env_name in SUPPORTED_V5_ENVS:
            with self.subTest(env_name=env_name):
                env, obs, next_obs, reward, done, info, expected_shape = _exercise_env(env_name, "cuda:0")
                self.assertEqual(tuple(obs.shape), expected_shape)
                self.assertEqual(tuple(next_obs.shape), expected_shape)
                self.assertEqual(tuple(reward.shape), (env.num_envs,))
                self.assertEqual(tuple(done.shape), (env.num_envs,))
                self.assertEqual(tuple(info["ale.lives"].shape), (env.num_envs,))
                self.assertTrue(bool(torch.isfinite(reward).all().item()))


if __name__ == "__main__":
    unittest.main()
