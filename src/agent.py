import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

def get_analysis_agent(df):
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL", "openrouter/auto") 
    base_url = "https://openrouter.ai/api/v1"

    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name=model,
        temperature=0,
        max_tokens=1500,
        timeout=90,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Ultimate Data Analyzer",
        }
    )

    PREFIX = """You are a Python expert. You are working with a pandas dataframe named `df`.
    You MUST use the `python_repl_ast` tool for ANY calculations.
    
    Step-by-step:
    1. Look at `df.columns` to see what data you have.
    2. Write and run code to get the answer.
    3. Formulate the Final Answer in Russian.
    
    If you get an error, fix your code and try again. 
    Never ask the user for more data, it's already in the `df` variable."""

    return create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True,
        allow_dangerous_code=True,
        agent_type="zero-shot-react-description", 
        prefix=PREFIX,
        max_iterations=10,
        max_execution_time=120,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )

def run_secure_query(agent, query: str):
    return agent.invoke({"input": f"{query}. ОТВЕТЬ НА РУССКОМ ЯЗЫКЕ."})