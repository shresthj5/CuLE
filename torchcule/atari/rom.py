"""CuLE (CUda Learning Environment module)."""

import importlib.resources as resources
import os
import re
from pathlib import Path

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility fallback
    import gym  # type: ignore[no-redef]

try:
    import ale_py
except ImportError:  # pragma: no cover - compatibility fallback
    ale_py = None

try:
    import atari_py
except ImportError:  # pragma: no cover - compatibility fallback
    atari_py = None

from torchcule_atari import AtariRom


def _register_ale_envs():
    if ale_py is None or not hasattr(gym, "register_envs"):
        return

    try:
        gym.register_envs(ale_py)
    except Exception:
        # The envs may already be registered.
        pass


def _camel_to_snake(name):
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+", name)
    return "_".join(part.lower() for part in parts if part)


def _canonical_game_name(env_name):
    _register_ale_envs()

    try:
        return gym.spec(env_name).kwargs["game"]
    except Exception:
        pass

    match = re.match(r"^(?:ALE/)?(?P<game>[A-Za-z0-9]+?)(?:NoFrameskip)?(?:-v\d+)?$", env_name)
    if match is None:
        raise ValueError(f"Unsupported Atari environment name: {env_name}")

    return _camel_to_snake(match.group("game"))


def _candidate_rom_paths(game_name):
    candidates = []

    rom_dir = os.environ.get("ALE_ROM_DIR")
    if rom_dir:
        candidates.append(Path(rom_dir) / f"{game_name}.bin")

    if atari_py is not None:
        for legacy_name in {game_name, game_name.replace("_", "")}:
            try:
                candidates.append(Path(atari_py.get_game_path(legacy_name)))
            except Exception:
                continue

    if ale_py is not None:
        rom_root = resources.files("ale_py.roms")
        candidates.append(Path(rom_root / f"{game_name}.bin"))

    return candidates

class Rom(AtariRom):

    def __init__(self, env_name):
        game_name = _canonical_game_name(env_name)

        for game_path in _candidate_rom_paths(game_name):
            if game_path.exists():
                try:
                    super(Rom, self).__init__(os.fspath(game_path), game_name)
                except RuntimeError as exc:
                    if f"Unsupported game name {game_name}" in str(exc):
                        raise ValueError(
                            "CuLE does not currently support %s (canonical game %s). "
                            "The ROM is installed, but CuLE lacks the game-specific "
                            "metadata and reward/life/action handlers required to run it."
                            % (env_name, game_name)
                        ) from exc
                    raise
                return

        raise IOError(
            "Requested environment (%s) does not resolve to an installed ROM. "
            "Expected a ROM such as %s.bin in ale-py or ALE_ROM_DIR."
            % (env_name, game_name)
        )

    def __repr__(self):
        return 'Name       : {}\n'\
               'Controller : {}\n'\
               'Swapped    : {}\n'\
               'Left Diff  : {}\n'\
               'Right Diff : {}\n'\
               'Type       : {}\n'\
               'Display    : {}\n'\
               'ROM Size   : {}\n'\
               'RAM Size   : {}\n'\
               'MD5        : {}\n'\
               .format(self.game_name(),
                       'Paddles' if self.use_paddles() else 'Joystick',
                       'Yes' if self.swap_paddles() or self.swap_ports() else 'No',
                       'B' if self.player_left_difficulty_B() else 'A',
                       'B' if self.player_right_difficulty_B() else 'A',
                       self.type(),
                       'NTSC' if self.is_ntsc() else 'PAL',
                       self.rom_size(),
                       self.ram_size(),
                       self.md5())
