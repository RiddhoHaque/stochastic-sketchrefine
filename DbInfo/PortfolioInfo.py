from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from ScenarioGenerator.PorfolioScenarioGenerator.GainScenarioGenerator import GainScenarioGenerator


class PortfolioInfo(DbInfo):

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
            return GainScenarioGenerator
        raise Exception('Unknown Attribute')
    
    @staticmethod
    def is_deterministic_attribute(
        attribute: str
    ) -> bool:
        return (attribute in \
            PortfolioInfo.get_deterministic_attributes())
    
    @staticmethod
    def get_diameter_threshold(
        attribute: str
    ) -> float:
        if attribute == 'gain':
            return Hyperparameters.DIAMETER_THRESHOLD_PORTFOLIO_GAIN
        if attribute == 'price':
            return Hyperparameters.DIAMETER_THRESHOLD_PORTFOLIO_PRICE