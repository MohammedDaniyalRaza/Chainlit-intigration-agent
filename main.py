from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import chainlit as cl
import requests

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

# 🟢 Fetch Daniyal's profile
def fetch_daniyal_profile():
    try:
        response = requests.get("https://mohammeddaniyalraza.vercel.app/api/profile")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Profile data not found."}
    except Exception as e:
        return {"error": str(e)}

profile_data = fetch_daniyal_profile()

profile_agent = Agent(
    name="Daniyal's Profile",
    instructions=f"""
You are a personal assistant chatbot for Mohammed Daniyal Raza.

The following is Daniyal's profile data fetched via API:
{profile_data}

✅ Your job is to answer any questions related to Daniyal's:
- Skills
- Experience
- Education
- Contact Info
- Portfolio links
- Projects
- Services he offers
- Everything!!!

🚫 If someone asks about anything else (not related to Daniyal), politely decline and say:
"I'm only here to answer questions about Mohammed Daniyal Raza."
"""
)


backend_agent = Agent(
    name="Back-end Developer",
    instructions="""
You are an expert backend developer. 
✅ Only answer questions strictly related to backend development (e.g., APIs, databases, server logic, authentication, backend frameworks, etc.).
🚫 Do NOT answer frontend (UI/UX) or any unrelated topics.
If asked something outside your domain, politely respond that it's not your area of expertise.
"""
)


frontend_agent = Agent(
    name="Front-end Developer",
    instructions="""
You are a professional frontend developer. 
✅ You only respond to questions related to frontend technologies (e.g., HTML, CSS, JavaScript, React, animations, UI, UX).
🚫 Do NOT answer anything related to backend, APIs, databases, or unrelated topics.
If asked something outside your scope, clearly say it's not your area.
"""
)


web_dev_agent = Agent(
    name="Web",
    instructions=f"""
You are the main web coordinator named "Web".

🎯 Your job is to manage incoming questions:
- If a question is related to **frontend** (HTML, CSS, JavaScript, React, UI/UX), hand it off to the **Front-end Developer agent**.
- If a question is related to **backend** (APIs, databases, server-side logic, authentication), hand it off to the **Back-end Developer agent**.
if a question about daniyal you can hand off to profile agent!

🛑 If the question is about anything else (e.g., AI, SEO, freelancing, design, or general chit-chat), you can respond directly in a helpful and friendly way.
""",
    handoffs=[frontend_agent, backend_agent, profile_agent]
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
        web_dev_agent,
        input=history,
        run_config=config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role" : "assistant" , "content" : result.final_output})

    cl.user_session.set("history", history)

    # await cl.Message(content=result.final_output).send()



