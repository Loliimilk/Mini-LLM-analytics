import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

def get_analysis_agent(df):
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL", "anthropic/claude-3.5-sonnet") 
    base_url = "https://openrouter.ai/api/v1"

    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name=model,
        temperature=0,
        max_tokens=2000,
        timeout=90,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Ultimate Data Analyzer",
        }
    )

# Измененная часть PREFIX в src/agent.py
    PREFIX = """You are a Python expert and Data Scientist. You work with a pandas dataframe named `df`.
    
    RULES FOR GRAPHICS:
    1. If the user asks for MULTIPLE charts or plots, use `plt.subplots()` to create a grid (e.g., 1x2 or 2x2) so ALL of them are visible in one image.
    2. ALWAYS save the final result to 'temp_chart.png' using `plt.savefig('temp_chart.png', bbox_inches='tight')`.
    3. Use `plt.close()` after saving.
    4. Make sure titles and labels are clear.
    
    STRICT RULE: Use the `python_repl_ast` tool for ALL calculations and plotting.
    Respond in Russian."""

    return create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True,
        allow_dangerous_code=True,
        agent_type="zero-shot-react-description", 
        prefix=PREFIX,
        max_iterations=10,
        handle_parsing_errors=True
    )

def run_secure_query(agent, query: str):
    return agent.invoke({"input": f"{query}. ОТВЕТЬ НА РУССКОМ ЯЗЫКЕ."})