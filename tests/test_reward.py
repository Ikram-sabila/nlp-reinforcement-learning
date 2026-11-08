from nlp_rl.rewards.token_match_reward import TokenMatchReward


def test_correct_token_receives_positive_reward():
    reward = TokenMatchReward("i like cats")

    assert reward.calculate_step("i", 0) == 1.0


def test_incorrect_token_receives_penalty():
    reward = TokenMatchReward("i like cats")

    assert reward.calculate_step("dogs", 0) == -0.5


def test_exact_sequence_receives_terminal_reward():
    reward = TokenMatchReward("i like cats")

    assert reward.calculate_terminal(["i", "like", "cats"]) == 3.0


def test_incorrect_sequence_receives_terminal_penalty():
    reward = TokenMatchReward("i like cats")

    assert reward.calculate_terminal(["i", "love", "cats"]) == -1.0