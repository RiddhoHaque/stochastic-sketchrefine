from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from ScenarioGenerator.PorfolioScenarioGenerator.GARCHFHSGainScenarioGenerator import GARCHFHSGainScenarioGenerator


class GarchPortfolioInfo(DbInfo):

    @staticmethod
    def get_deterministic_attributes():
        return ['price']
    
    @staticmethod
    def get_stochastic_attributes():
        return ['gain']
    
    @staticmethod
    def get_variable_generator_function(
        attribute: str) -> ScenarioGenerator:
        if attribute == 'gain':
            return GARCHFHSGainScenarioGenerator
        raise Exception('Unknown Attribute')
    
    @staticmethod
    def is_deterministic_attribute(
        attribute: str
    ) -> bool:
        return (attribute in \
            GarchPortfolioInfo.get_deterministic_attributes())
    
    @staticmethod
    def get_diameter_threshold(
        attribute: str
    ) -> float:
        if attribute == 'gain':
            return Hyperparameters.DIAMETER_THRESHOLD_GARCH_GAIN
        if attribute == 'price':
            return Hyperparameters.DIAMETER_THRESHOLD_GARCH_PRICE
        
    @staticmethod
    def has_inter_tuple_correlations() -> bool:
        return True