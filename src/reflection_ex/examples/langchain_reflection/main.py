"""LangChain example of the reflection pattern:
   This is basically a sequential prompt chain utilizing the concepts
   of reflection.
   Source = Agent Design Patterns by Antonio Gulli
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


MODEL = "gpt-5-mini"


def build_reflection_chain(llm: BaseChatModel):
    """Build the producer, reviewer, and refinement steps."""

    producer_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Write a short product description. Keep it simple and don't invent features.",
                ),
                ("human", "Product details:\n{product_details}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    reviewer_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful editor. Give specific suggestions about clarity, "
                    "conciseness, and appeal. Don't invent product features.",
                ),
                (
                    "human",
                    "Review this product description:\n{initial_description}",
                ),
            ]
        )
        | llm
        | StrOutputParser()
    )

    refinement_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful content editor. Improve the draft using the critique. "
                    "Preserve the product facts and don't invent features.",
                ),
                (
                    "human",
                    "Revise this product description.\n\n"
                    "Original product details:\n{product_details}\n\n"
                    "Draft:\n{initial_description}\n\n"
                    "Critique:\n{critique}",
                ),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(initial_description=producer_chain)
        | RunnablePassthrough.assign(critique=reviewer_chain)
        | refinement_chain
    )


async def run_reflection_example(reflection_chain, product_details: str) -> None:
    """Run the chain once and print its final description."""

    result = await reflection_chain.ainvoke(
        {"product_details": product_details}
    )
    print("\nFinal refined product description:\n")
    print(result)


def main() -> None:
    """Set up the model and run this example."""

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file.")

    llm = ChatOpenAI(model=MODEL, temperature=0.7)
    reflection_chain = build_reflection_chain(llm)

    product_details = (
        "A travel-friendly mug that keeps coffee and tea hot and can be "
        "controlled with a smartphone app."
    )
    asyncio.run(run_reflection_example(reflection_chain, product_details))


if __name__ == "__main__":
    main()
