import os
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI , OpenAIChatCompletionsModel # type: ignore
from agents.run import RunConfig  # type: ignore
from dotenv import load_dotenv, find_dotenv  # type: ignore
from openai.types.responses import ResponseTextDeltaEvent  # type: ignore


load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True

)


agent = Agent(
    name = "assistant",
    instructions= "my assistant",
    model = model
)


@cl.on_chat_start
async def handle_histry():
    cl.user_session.set("history", [])
    await cl.Message(content="👋 Hi! I'm your assistant. How can I help you?").send()

@cl.on_message
async def handle(message:cl.Message):
    history = cl.user_session.get("history") or []

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role":"user", "content":message.content})

    result =   Runner.run_streamed(
        agent,
        input=history,
        run_config=config
        
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data , ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
            
    history.append({"role":"assistant", "content":result.final_output} )
    cl.user_session.set("history",history)

    





