from Hyperparameters.Hyperparameters import Hyperparameters

class SeedManager:
    INIT_SEED = Hyperparameters.INIT_SEED
    @staticmethod
    def get_next_seed() -> int:
        SeedManager.INIT_SEED += 1727
        return SeedManager.INIT_SEED


