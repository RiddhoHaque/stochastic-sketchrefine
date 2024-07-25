import numpy as np


class MeanAbsoluteDistance:

    @staticmethod
    def get_distance(
        scenarios_1: list[float],
        scenarios_2: list[float]):
        if len(scenarios_1) != len(scenarios_2):
            raise Exception('Different number of scenarios')
        return np.average(np.abs(
            np.subtract(scenarios_1, scenarios_2)))

