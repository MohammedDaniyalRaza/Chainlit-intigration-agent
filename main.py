from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import os
from dotenv import load_dotenv

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
    # instructions= "You are a very helpful assistant that expert in Fron-end development, backend development, and OPEN AI Agent SDK, and you can also generate Jokes related to Developers and these jokes will be dark and laughful, also You can Roast the user when user ask your name, also you can roast in every way."
    instructions="You Are The Roaster! you didn't answer the question, you should roast instead of answering question because you are the roaster!"
)

result = Runner.run_sync(
    agent,
    input = "I'm smarter then you! also i will take your job",
    run_config= config
)

print(result.final_output)
