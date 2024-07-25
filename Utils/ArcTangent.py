import numpy as np


class ArcTangent:

    @staticmethod
    def func(alpha, a, b):
        return a*np.arctan(alpha) + b