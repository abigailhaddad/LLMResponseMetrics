from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage

chat = ChatAnthropic(model="claude-1")

messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
chat(messages)

response = chat(messages, temperature=0.7)