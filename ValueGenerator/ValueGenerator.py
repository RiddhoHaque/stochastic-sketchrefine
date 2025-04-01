from PgConnection.PgConnection import PgConnection
from Utils.Relation_Prefixes import Relation_Prefixes


class ValueGenerator:

    def __init__(self, relation,
                 base_predicate,
                 attribute):
        self.__relation = relation
        self.__base_predicate = \
            base_predicate
        self.__attribute = \
            attribute
        
    def get_values(self):
        sql_query = "SELECT " + \
            self.__attribute + \
            " FROM " + self.__relation
        if len(self.__base_predicate) > 0:
            sql_query += " WHERE " + \
                self.__base_predicate
        sql_query += " ORDER BY id;"
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()
    
    def get_values_from_partition(
        self, partition_id: int
    ):
        self.__relation = self.__relation +\
            ' AS r INNER JOIN ' + \
                Relation_Prefixes.PARTITION_RELATION_PREFIX +\
                    self.__relation + ' AS p ON r.id=p.tuple_id'

        if len(self.__base_predicate) > 0:
            self.__base_predicate += ' AND '
        self.__base_predicate += 'p.partition_id = ' + str(
            partition_id
        )

        return self.get_values()