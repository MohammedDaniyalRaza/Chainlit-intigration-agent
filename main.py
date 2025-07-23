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
    instructions="You are just helpful assistant! you can assist in roman urdu and english also."
)

@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(content="🤖 Hello, How Can I Assist You?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role" : "user", "content" : message.content})

    result = await Runner.run(
        agent,
        input=history,
        run_config=config
    )

    history.append({"role" : "assistant" , "content" : result.final_output})

    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()



