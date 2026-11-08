from nlp_rl.config import ExperimentConfig
from nlp_rl.environments.text_generation_env import (
    TextGenerationEnvironment,
)
from nlp_rl.rewards.token_match_reward import TokenMatchReward


def create_environment() -> TextGenerationEnvironment:
    config = ExperimentConfig()

    reward = TokenMatchReward(
        target_text=config.target_text,
    )

    return TextGenerationEnvironment(
        config=config,
        reward_function=reward,
    )


def test_reset_returns_beginning_token():
    env = create_environment()

    state = env.reset()

    assert state == [env.bos_id]
    assert env.generated_text == ""


def test_environment_generates_token():
    env = create_environment()
    env.reset()

    token_id = env.token_to_id["i"]

    state, reward, done = env.step(token_id)

    assert state == [env.bos_id, token_id]
    assert reward == 1.0
    assert done is False
    assert env.generated_text == "i"


def test_exact_sequence_receives_terminal_reward():
    env = create_environment()
    env.reset()

    for token in ["i", "like", "cats"]:
        _, reward, done = env.step(env.token_to_id[token])

    assert done is False
    assert reward == 1.0

    _, reward, done = env.step(env.eos_id)

    assert done is True
    assert reward == 3.0
    assert env.generated_text == "i like cats"