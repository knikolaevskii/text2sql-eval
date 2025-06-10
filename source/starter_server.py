from premsql.playground import AgentServer
from premsql.agents import BaseLineAgent
from premsql.generators import Text2SQLGeneratorOllama
from premsql.agents.tools import SimpleMatplotlibTool
from premsql.executors import ExecutorUsingLangChain

text2sql_model = Text2SQLGeneratorOllama(
    model_name="anindya/prem1b-sql-ollama-fp116",
    experiment_name="ollama",
    type="test"
)

analyser_plotter_model = Text2SQLGeneratorOllama(
    model_name="llama3.2:1b",
    experiment_name="ollama",
    type="test"
)

db_connection_uri = "sqlite:////Users/kirillnikolaevskii/Library/Caches/premsql/kaggle/students.sqlite"
baseline = BaseLineAgent(
    session_name="students",                # An unique session name must be put
    db_connection_uri=db_connection_uri,        # DB which needs to connect for Text to SQL 
    specialized_model1=text2sql_model,          # This referes to the Text to SQL model
    specialized_model2=analyser_plotter_model,  # This refers to any model other than Text to SQL
    executor=ExecutorUsingLangChain(),          # Which DB executor to use
    auto_filter_tables=False,                   # Whether to filter tables before Text to SQL
    plot_tool=SimpleMatplotlibTool()            # Matplotlib Tool which will be used by plotter worker
)

agent_server = AgentServer(agent=baseline, port=7824)
agent_server.launch()