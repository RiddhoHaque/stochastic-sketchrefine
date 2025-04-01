from PgConnection.PgConnection import PgConnection
from Utils.Relation_Prefixes import Relation_Prefixes

class RepresentativeValueGenerator:

    def __init__(self, relation: str,
                 base_predicate: str,
                 attribute: str,
                 duplicate_vector: list[int]):
        self.__relation = relation
        self.__base_predicate = base_predicate
        self.__attribute = attribute
        self.__duplicate_vector = duplicate_vector
        
    
    def get_values(self):
        representative_relation = \
            Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX +\
                self.__relation
        if len(self.__base_predicate) > 0:
            self.__base_predicate += ' AND '
        self.__base_predicate += "r.attribute='" + self.__attribute + "'"
        sql = "SELECT t." + self.__attribute +\
            " FROM " + representative_relation + " AS r INNER JOIN " +\
            self.__relation + " AS t ON (r.representative_tuple_id=t." +\
            "id) WHERE " + self.__base_predicate + " ORDER BY r.partition_id;"
        PgConnection.Execute(sql)
        tuples = PgConnection.Fetch()

        values = []

        index = 0
        for tuple in tuples:
            for _ in range(self.__duplicate_vector[index]):
                values.append(tuple[0])
            index += 1
        
        return values