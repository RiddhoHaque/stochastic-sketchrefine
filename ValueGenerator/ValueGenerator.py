from PgConnection.PgConnection import PgConnection


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
        