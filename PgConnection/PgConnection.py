from configparser import ConfigParser
import psycopg2

class PgConnection:

    CONNECTION = None
    CURSOR = None

    @staticmethod
    def Config(filename='Data/database.ini',
               section = 'postgresql'):
        parser = ConfigParser()
        parser.read(filename)

        db_config = {}

        if section in parser:
            for key in parser[section]:
                db_config[key] = parser[section][key]
        
        return db_config
    
    @staticmethod
    def Connect_to_DB():
        db_config = PgConnection.Config()
        return psycopg2.connect(
            dbname = db_config['dbname'],
            user = db_config['user'],
            host = db_config['host'],
            password = db_config['password'],
            port = db_config['port']
        )

    @staticmethod 
    def Cursor():
        if PgConnection.CONNECTION is None:
             PgConnection.CONNECTION = PgConnection.Connect_to_DB()
        return PgConnection.CONNECTION.cursor()
    
    @staticmethod
    def Execute(sql: str):
        if PgConnection.CURSOR is None:
            PgConnection.CURSOR = PgConnection.Cursor()
        PgConnection.CURSOR.execute(sql)
    
    @staticmethod
    def Fetch():
        if PgConnection.CURSOR is None:
            return Exception()
        return PgConnection.CURSOR.fetchall()
    
    @staticmethod
    def Commit():
        if PgConnection.CONNECTION is None:
            return Exception()
        PgConnection.CONNECTION.commit()

