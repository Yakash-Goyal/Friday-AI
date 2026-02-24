from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Connect to your local LLaMA 3.1 model
model = OllamaLLM(model="llama3.1")

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer this clearly: {question}"
)

# Chain prompt and model together
chain = prompt | model

print("Local AI Assistant is running!")
print("Type 'quit' to exit")
print("=" * 40)

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chain.invoke({"question": user_input})
    print(f"AI: {response}")
    print()