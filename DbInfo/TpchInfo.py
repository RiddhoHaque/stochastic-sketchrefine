from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from ScenarioGenerator.TpchScenarioGenerators.PriceScenarioGenerator import PriceScenarioGenerator
from ScenarioGenerator.TpchScenarioGenerators.QuantityScenarioGenerator import QuantityScenarioGenerator


class TpchInfo(DbInfo):

    @staticmethod
    def get_deterministic_attributes():
        return ['tax']
    
    @staticmethod
    def get_stochastic_attributes():
        return ['price', 'quantity']
    
    @staticmethod
    def get_variable_generator_function(
        attribute: str) -> ScenarioGenerator:
        if attribute == 'price':
            return PriceScenarioGenerator
        
        if attribute == 'quantity':
            return QuantityScenarioGenerator
        
        raise Exception('Attribute Unknown')
    
    @staticmethod
    def is_deterministic_attribute(
        attribute: str
    ) -> bool:
        return (attribute in \
            TpchInfo.get_deterministic_attributes())
    
    @staticmethod
    def get_diameter_threshold(
        attribute: str
    ) -> float:
        if attribute == 'price':
            return Hyperparameters.DIAMETER_THRESHOLD_TPCH_PRICE
        if attribute == 'quantity':
            return Hyperparameters.DIAMETER_THRESHOLD_TPCH_QUANTITY
        if attribute == 'tax':
            return Hyperparameters.DIAMETER_THRESHOLD_TPCH_TAX

    @staticmethod
    def has_inter_tuple_correlations() -> bool:
        return False