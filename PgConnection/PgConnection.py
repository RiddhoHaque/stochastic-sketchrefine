from configparser import ConfigParser
import time
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
            port = db_config['port'],
            keepalives = 1,
            keepalives_idle = 60,
            keepalives_interval = 10,
            keepalives_count = 5,
        )

    @staticmethod
    def Cursor():
        if PgConnection.CONNECTION is None:
             PgConnection.CONNECTION = PgConnection.Connect_to_DB()
        return PgConnection.CONNECTION.cursor()

    @staticmethod
    def _reconnect(max_attempts=5, wait_seconds=5):
        for attempt in range(1, max_attempts + 1):
            try:
                try:
                    if PgConnection.CONNECTION is not None:
                        PgConnection.CONNECTION.close()
                except Exception:
                    pass
                PgConnection.CONNECTION = PgConnection.Connect_to_DB()
                PgConnection.CURSOR = PgConnection.CONNECTION.cursor()
                print(f'  [PgConnection] Reconnected on attempt {attempt}.')
                return
            except Exception as e:
                print(f'  [PgConnection] Reconnect attempt {attempt} failed: {e}')
                if attempt < max_attempts:
                    time.sleep(wait_seconds)
        raise RuntimeError(
            f'[PgConnection] Could not reconnect after {max_attempts} attempts.')

    @staticmethod
    def Execute(sql: str):
        if PgConnection.CURSOR is None:
            PgConnection.CURSOR = PgConnection.Cursor()
        try:
            PgConnection.CURSOR.execute(sql)
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            print(f'  [PgConnection] Connection lost. Reconnecting...')
            PgConnection._reconnect()
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
