"""
Example of using LangChain models with browser-use.

This example demonstrates how to:
1. Wrap a LangChain model with ChatLangchain
2. Use it with a browser-use Agent
3. Run a simple web automation task

@file purpose: Example usage of LangChain integration with browser-use
"""

import asyncio
import os

from langchain_openai import ChatOpenAI  # pyright: ignore

from browser_use import Agent
from chat import ChatLangchain

API_KEY = os.getenv("API_KEY", "")
LOCAL_URL = os.getenv("BASE_URL", "")
MODELO = 'qwen3-coder:30b'

async def main():
	"""Basic example using ChatLangchain with OpenAI through LangChain."""

	langchain_model = ChatOpenAI(
		base_url=LOCAL_URL,
        model=MODELO,
        api_key=API_KEY,
		temperature=0.0,
		max_tokens=4096,
	)

	llm = ChatLangchain(chat=langchain_model)

	task = "Go to google.com and search for 'browser automation with Python'"

	agent = Agent(
		task=task,
		llm=llm,
	)

	print(f'🚀 Starting task: {task}')
	print(f'🤖 Using model: {llm.name} (provider: {llm.provider})')

	history = await agent.run()

	print(f'✅ Task completed! Steps taken: {len(history.history)}')

	if history.final_result():
		print(f'📋 Final result: {history.final_result()}')

		return history


if __name__ == '__main__':
	print('🌐 Browser-use LangChain Integration Example')
	print('=' * 45)

	asyncio.run(main())