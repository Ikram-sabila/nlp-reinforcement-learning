import torch
from collections import Counter

from nlp_rl.config import ExperimentConfig
from nlp_rl.experiments.compare_reinforce import REINFORCEComparison


def main() -> None:
    config = ExperimentConfig()

    experiment = REINFORCEComparison(config=config)
    experiment.compare()


if __name__ == "__main__":
    main()