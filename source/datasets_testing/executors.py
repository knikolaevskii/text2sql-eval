from premsql.executors import SQLiteExecutor

# Instantiate the executor
executor = SQLiteExecutor()

# Set a sample dataset path
db_path = "spider/database/cinema/cinema.sqlite"
sql = 'SELECT * FROM cinema;'


# execute the SQL
result = executor.execute_sql(
    sql=sql,
    dsn_or_db_path=db_path
)

print(result)

from premsql.executors import ExecutorUsingLangChain

executor = ExecutorUsingLangChain()
result = executor.execute_sql(
    sql=sql,
    dsn_or_db_path=db_path
)
print("--------------------------------")
print(result)