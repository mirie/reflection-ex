from google.adk.agents import Agent

MODEL = "gemini-3.5-flash-lite"

root_agent = Agent(
    name="reflection_agent",
    model=MODEL,
    instruction=(
        "Answer the user's request. Review your draft for accuracy and completeness, "
        "then provide an improved final answer."
    ),
)
