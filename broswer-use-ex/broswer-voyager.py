import sys
print(sys.version)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig

import asyncio
import os

import logging
from datetime import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

history_dir = "history"
os.makedirs(history_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_dir, f"ex-2_log_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        # logging.StreamHandler()  # Optional: also print to console
    ]
)


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
	raise ValueError('GEMINI_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

browser = Browser(
	config=BrowserConfig(
		new_context_config=BrowserContextConfig(
			viewport_expansion=0,
		)
	)
)

TASK = """
Find and book a hotel in Ujjain, India with suitable accommodations for three adults offering free cancellation 
for the dates of May 20-23, 2025. on https://www.booking.com/. the budget for the per night is between 3000-5000 and pet friendly. 
DONOT exceed the per night budget 
"""


async def main():
	agent = Agent(
		task=TASK,
		llm=llm,
		browser=browser,
		validate_output=True,
		enable_memory=False,
	)
	history = await agent.run(max_steps=50)
	save_filename = os.path.join(history_dir, f"ex-2_history_{timestamp}.json")
	history.save_to_file(save_filename)


if __name__ == '__main__':
	asyncio.run(main())