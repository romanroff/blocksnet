from pathlib import Path
from blocksnet.machine_learning.strategy.catboost import CatBoostClassificationStrategy

CURRENT_DIRECTORY = Path(__file__).parent
ARTIFACTS_DIRECTORY = str(CURRENT_DIRECTORY / "artifacts")


def get_default_strategy() -> CatBoostClassificationStrategy:
    strategy = CatBoostClassificationStrategy()
    strategy.load(ARTIFACTS_DIRECTORY)
    return strategy
