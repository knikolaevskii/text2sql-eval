import json
import time
import pandas as pd
from sqlalchemy import create_engine, text
from func_timeout import func_timeout
from typing import Optional

class PostgresExecutor:
    def execute_sql(self, sql: str, dsn_or_db_path: Optional[str] = None) -> dict:
        """
        Executes a SQL query on a PostgreSQL database using credentials from file.

        :param sql: SQL query to execute
        :param dsn_or_db_path: Path to a JSON file containing DB credentials:
            {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "password": "postgres",
            }

        :return: dict with result summary:
            {
                "result": result (raw object or error),
                "result_df": DataFrame or None,
                "error": error message or None,
            }
        """

        result = None
        error = None
        df = None

        dsn_or_db_path = "/Users/kirillnikolaevskii/Desktop/prem/source/utils/creds.json"

        # Load credentials
        try:
            with open(dsn_or_db_path, "r") as f:
                creds = json.load(f)

            print(creds)
            db_name = "academic"
        

            db_url = (
                f"postgresql://{creds['user']}:{creds['password']}"
                f"@{creds['host']}:{creds['port']}/{db_name}"
            )
            print("good")

            engine = create_engine(db_url)

            print("bad")

            with engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                df = func_timeout(10.0, pd.read_sql_query, args=(sql, conn))
                result = df.to_dict(orient="records")
                print(result)

        except Exception as e:
            print("error", str(e))
            error = str(e)

        return {
            "result": result,
            "result_df": df,
            "error": error,
        }
