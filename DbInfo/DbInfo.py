from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator


class DbInfo:

    @staticmethod
    def get_deterministic_attributes():
        return []

    @staticmethod
    def get_stochastic_attributes():
        return []
    
    @staticmethod
    def get_variable_generator_function(
        attribute: str
    ) -> ScenarioGenerator:
        return ScenarioGenerator
    
    @staticmethod
    def is_deterministic_attribute(
        attribute: str
    ) -> bool:
        return (attribute in \
            DbInfo.get_deterministic_attributes())
    
    @staticmethod
    def get_diameter_threshold(
        attribute: str
    ) -> float:
        return 0