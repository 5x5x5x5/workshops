import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "langchain-core", "langgraph", "langchain-openai"
)

app = modal.App("langgraph-react-agent", image=image)


@app.function(
    cpu=1,
    memory=1024,
    secrets=[modal.Secret.from_name("openai-secret")],
)
async def agent(request: str) -> str:
    """Run a LangGraph ReAct agent with math tools."""
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    @tool
    async def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @tool
    async def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    llm = ChatOpenAI(model="gpt-4o-mini")
    react_agent = create_react_agent(llm, [add, multiply])

    result = await react_agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    return result["messages"][-1].content


@app.local_entrypoint()
def main(request: str = "What is 12 * 7 plus 3?"):
    print(agent.remote(request))


# uv run modal run langgraph_react_agent.py --request "What is 12 * 7 plus 3?"
