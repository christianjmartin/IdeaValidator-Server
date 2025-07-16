from langchain_tavily import TavilySearch
from langchain.tools import tool 
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilySearch(
    max_results=8,
    topic="general",
)

@tool()
def generate_roadmap(idea: str, important_context: str) -> str:
    """Creates a unique and detailed roadmap to help the user build their idea. Use both the idea and context to make it specific and actionable."""
    return f"Generate roadmap for idea: {idea}"


tools = [search_tool, generate_roadmap]