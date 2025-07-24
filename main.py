from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
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

    msg = cl.Message(content="")
    await msg.send()

    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role" : "assistant" , "content" : result.final_output})

    cl.user_session.set("history", history)

    # await cl.Message(content=result.final_output).send()



