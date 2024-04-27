import unittest
from StochasticPackageQuery.Parser.Parser import Parser
from Utils.RelationalOperators import RelationalOperators


class ParserUnitTest(unittest.TestCase):

    def test_parser_construction(self):
        Parser()

    def test_query_preprocessing(self):
        parser = Parser()
        query = parser.preprocess(self.base_query_lines)
        self.assertEqual(query, self.expected_preprocessed_query)

    def test_base_query_parsing(self):
        parser = Parser()
        query = parser.parse(self.base_query_lines)
        self.assertEqual(query.get_projected_attributes(), '*')
        self.assertEqual(query.get_package_alias(), 'portfolio')
        self.assertEqual(query.get_relation(), 'stock_investments')
        self.assertEqual(query.get_relation_alias(), 's')
        self.assertEqual(query.get_base_predicate(), "category='technology' ")
        self.assertEqual(len(query.get_constraints()), 4)
        self.assertTrue(query.get_constraints()[0].is_repeat_constraint())
        self.assertEqual(query.get_constraints()[0].get_repetition_limit(), 1)
        self.assertTrue(query.get_constraints()[1].is_package_size_constraint())
        self.assertEqual(query.get_constraints()[1].get_inequality_sign(),
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[1].get_package_size_limit(),
                         30)
        self.assertTrue(query.get_constraints()[2].is_deterministic_constraint())
        self.assertEqual(query.get_constraints()[2].get_inequality_sign(),
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[2].get_attribute_name(),
                         'price')
        self.assertEqual(query.get_constraints()[2].get_sum_limit(), 5000)
        self.assertTrue(query.get_constraints()[3].is_risk_constraint())
        self.assertTrue(query.get_constraints()[3].is_var_constraint())
        self.assertEqual(query.get_constraints()[3].get_attribute_name(),
                        'gain')
        self.assertEqual(query.get_constraints()[3].get_inequality_sign(),
                        RelationalOperators.GREATER_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[3].get_sum_limit(), -50)
        self.assertEqual(query.get_constraints()[3].get_probability_threshold(),
                         0.95)
    
    def test_without_aliases(self):
        query_lines = []
        for line in self.base_query_lines:
            query_lines.append(line)
        query_lines[0] = "SELECT PACKAGE(*)"
        query_lines[1] = "FROM Stock_Investments"
        parser = Parser()
        query = parser.parse(query_lines)
        self.assertEqual(query.get_projected_attributes(), '*')
        self.assertEqual(query.get_relation(), 'stock_investments')
        self.assertEqual(query.get_base_predicate(), "category='technology' ")
        self.assertEqual(len(query.get_constraints()), 4)
        self.assertTrue(query.get_constraints()[0].is_repeat_constraint())
        self.assertEqual(query.get_constraints()[0].get_repetition_limit(), 1)
        self.assertTrue(query.get_constraints()[1].is_package_size_constraint())
        self.assertEqual(query.get_constraints()[1].get_inequality_sign(),
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[1].get_package_size_limit(),
                         30)
        self.assertTrue(query.get_constraints()[2].is_deterministic_constraint())
        self.assertEqual(query.get_constraints()[2].get_inequality_sign(),
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[2].get_attribute_name(),
                         'price')
        self.assertEqual(query.get_constraints()[2].get_sum_limit(), 5000)
        self.assertTrue(query.get_constraints()[3].is_risk_constraint())
        self.assertTrue(query.get_constraints()[3].is_var_constraint())
        self.assertEqual(query.get_constraints()[3].get_attribute_name(),
                        'gain')
        self.assertEqual(query.get_constraints()[3].get_inequality_sign(),
                        RelationalOperators.GREATER_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[3].get_sum_limit(), -50)
        self.assertEqual(query.get_constraints()[3].get_probability_threshold(),
                         0.95)

    def test_without_selection_predicate(self):
        query_lines = []
        for idx in range(len(self.base_query_lines)):
            if idx == 3:
                continue
            query_lines.append(self.base_query_lines[idx])
        parser = Parser()
        query = parser.parse(query_lines)
        self.assertEqual(query.get_projected_attributes(), '*')
        self.assertEqual(query.get_package_alias(), 'portfolio')
        self.assertEqual(query.get_relation(), 'stock_investments')
        self.assertEqual(query.get_relation_alias(), 's')
        self.assertEqual(len(query.get_constraints()), 4)
        self.assertTrue(query.get_constraints()[0].is_repeat_constraint())
        self.assertEqual(query.get_constraints()[0].get_repetition_limit(), 1)
        self.assertTrue(query.get_constraints()[1].is_package_size_constraint())
        self.assertEqual(query.get_constraints()[1].get_inequality_sign(),
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[1].get_package_size_limit(),
                         30)
        self.assertTrue(query.get_constraints()[2].is_deterministic_constraint())
        self.assertEqual(query.get_constraints()[2].get_inequality_sign(),
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[2].get_attribute_name(),
                         'price')
        self.assertEqual(query.get_constraints()[2].get_sum_limit(), 5000)
        self.assertTrue(query.get_constraints()[3].is_risk_constraint())
        self.assertTrue(query.get_constraints()[3].is_var_constraint())
        self.assertEqual(query.get_constraints()[3].get_attribute_name(),
                        'gain')
        self.assertEqual(query.get_constraints()[3].get_inequality_sign(),
                        RelationalOperators.GREATER_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[3].get_sum_limit(), -50)
        self.assertEqual(query.get_constraints()[3].get_probability_threshold(),
                         0.95)

    def test_without_repeat_constraint(self):
        query_lines = []
        for idx in range(len(self.base_query_lines)):
            if idx == 2:
                continue
            query_lines.append(self.base_query_lines[idx])
        parser = Parser()
        query = parser.parse(query_lines)
        self.assertEqual(query.get_projected_attributes(), '*')
        self.assertEqual(query.get_package_alias(), 'portfolio')
        self.assertEqual(query.get_relation(), 'stock_investments')
        self.assertEqual(query.get_relation_alias(), 's')
        self.assertEqual(query.get_base_predicate(), "category='technology' ")
        self.assertEqual(len(query.get_constraints()), 3)
        self.assertTrue(query.get_constraints()[0].is_package_size_constraint())
        self.assertEqual(query.get_constraints()[0].get_inequality_sign(),
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[0].get_package_size_limit(),
                         30)
        self.assertTrue(query.get_constraints()[1].is_deterministic_constraint())
        self.assertEqual(query.get_constraints()[1].get_inequality_sign(),
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[1].get_attribute_name(),
                         'price')
        self.assertEqual(query.get_constraints()[1].get_sum_limit(), 5000)
        self.assertTrue(query.get_constraints()[2].is_risk_constraint())
        self.assertTrue(query.get_constraints()[2].is_var_constraint())
        self.assertEqual(query.get_constraints()[2].get_attribute_name(),
                        'gain')
        self.assertEqual(query.get_constraints()[2].get_inequality_sign(),
                        RelationalOperators.GREATER_THAN_OR_EQUAL_TO)
        self.assertEqual(query.get_constraints()[2].get_sum_limit(), -50)
        self.assertEqual(query.get_constraints()[2].get_probability_threshold(),
                         0.95)

    def main(self):
        self.base_query_lines = [
            "SELECT PACKAGE(*) AS Portfolio",
            "FROM Stock_Investments AS S",
            "REPEAT 1",
            "WHERE Category='Technology'",
            "SUCH THAT COUNT(*) <= 30 AND",
            "SUM(Price) <= 5000 AND",
            "SUM(Gain) >= -50 WITH PROBABILITY >= 0.95",
            "MAXIMIZE EXPECTED SUM(Gain)",
        ]
        self.expected_preprocessed_query = \
            " select package(*) as portfolio" \
            " from stock_investments as s" \
            " repeat 1" \
            " where category='technology'" \
            " #such that count(*) <= 30 and" \
            " sum(price) <= 5000 and" \
            " sum(gain) >= -50 with probability >= 0.95" \
            " maximize expected sum(gain)"
        self.test_parser_construction()
        self.test_query_preprocessing()
        self.test_base_query_parsing()
        self.test_without_aliases()
        self.test_without_selection_predicate()