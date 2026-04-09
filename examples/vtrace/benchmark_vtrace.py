import re
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility fallback
    import gym  # type: ignore[no-redef]

try:
    import ale_py
except ImportError:  # pragma: no cover - compatibility fallback
    ale_py = None

import os


def _register_ale_envs():
    if ale_py is None or not hasattr(gym, 'register_envs'):
        return

    try:
        gym.register_envs(ale_py)
    except Exception:
        pass


def atari_games():
    _register_ale_envs()
    pattern = re.compile(r'^ALE\/\w+-v5$')

    registry = gym.registry.values() if hasattr(gym, 'registry') else gym.envs.registry.all()
    return [env_spec.id for env_spec in registry if pattern.match(env_spec.id)]

env_names = atari_games()
env_names.remove('ALE/Qbert-v5')
env_names.remove('ALE/ElevatorAction-v5')
env_names.remove('ALE/Defender-v5')
num_ales_list = [1024, 2048, 16, 4096] #[1, 32, 64, 128, 256, 512, 1024, 2048, 4096]

for num_ales in num_ales_list:
    for env_name in env_names:

        if num_ales < 1025:
            os.system('python vtrace_main.py --benchmark --num-ales ' + str(num_ales) + ' --env-name ' + env_name + ' --num-steps 5 --num-minibatches 1 --num-steps-per-update 5 --normalize --use-openai')
        os.system('python vtrace_main.py --benchmark --num-ales ' + str(num_ales) + ' --env-name ' + env_name + ' --num-steps 5 --num-minibatches 1 --num-steps-per-update 5 --normalize')
        os.system('python vtrace_main.py --benchmark --num-ales ' + str(num_ales) + ' --env-name ' + env_name + ' --num-steps 5 --num-minibatches 1 --num-steps-per-update 5 --normalize --use-cuda-env')
