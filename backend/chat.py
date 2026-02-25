from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

model = OllamaLLM(model="llama3.1")

chat_history=[]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Friday, a helpful private AI assistant. You remember everything said in this conversation."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | model

print("Local AI Assistant is running!")
print("Type 'quit' to exit")
print("=" * 40)

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chain.invoke({"question": user_input, "chat_history" : chat_history})

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")
    print()