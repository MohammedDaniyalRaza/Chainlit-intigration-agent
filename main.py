from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()


apiKey = os.getenv("GEMINI_API_KEY")


externalClient = AsyncOpenAI(
    api_key=apiKey,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client= externalClient
)

config = RunConfig(
    model=model,
    model_provider=externalClient,
    tracing_disabled=True
)

agent = Agent(
    name= "WebDev",
    instructions="You Are The Roaster! you didn't answer the question, you should roast instead of answering question because you are the roaster!"
)

@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(
        agent,
        input=message.content,
        run_config=config
    )
    await cl.Message(content=result.final_output).send()



