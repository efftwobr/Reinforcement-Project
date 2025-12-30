from gymnasium.envs.registration import register
import wordle

# Classic
# ----------------------------------------

register(
    id="WordleEnv10-v0",
    entry_point="wordle:WordleEnv10",
    max_episode_steps=200,
)

register(
    id="WordleEnv100-v0",
    entry_point="wordle:WordleEnv100",
    max_episode_steps=500,
)

register(
    id="WordleEnv100OneAction-v0",
    entry_point="wordle:WordleEnv100OneAction",
    max_episode_steps=500,
)

register(
    id="WordleEnv100TwoAction-v0",
    entry_point="wordle:WordleEnv100TwoAction",
    max_episode_steps=500,
)

register(
    id="WordleEnv100FullAction-v0",
    entry_point="wordle:WordleEnv100FullAction",
    max_episode_steps=500,
)

register(
    id="WordleEnv100WithMask-v0",
    entry_point="wordle:WordleEnv100WithMask",
    max_episode_steps=500,
)

register(
    id="WordleEnv1000-v0",
    entry_point="wordle:WordleEnv1000",
    max_episode_steps=500,
)

register(
    id="WordleEnv1000WithMask-v0",
    entry_point="wordle:WordleEnv1000WithMask",
    max_episode_steps=500,
)

register(
    id="WordleEnv1000FullAction-v0",
    entry_point="wordle:WordleEnv1000FullAction",
    max_episode_steps=500,
)

register(
    id="WordleEnvFull-v0",
    entry_point="wordle:WordleEnvFull",
    max_episode_steps=500,
)

register(
    id="WordleEnvReal-v0",
    entry_point="wordle:WordleEnvReal",
    max_episode_steps=500,
)

register(
    id="WordleEnvRealWithMask-v0",
    entry_point="wordle:WordleEnvRealWithMask",
    max_episode_steps=500,
)
