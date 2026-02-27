from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# loading and processing the pdf
def load_and_index_pdf(pdf_path):
    print(f"Reading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")
    
    # Spliting  into chunks
    # Each chunk = 200 characters
    # chunk_overlap=200 means chunks share 200 characters with neighbors
    # overlap prevents losing context at chunk boundaries (sentence that falls between two chunks doesn't lose its meaning)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    
    # Converting chunks to embeddings and store them in FAISS(Facebook AI Similarity Search)(optimized specifically for finding nearest neighbor vectors extremely fast)
    print("Creating embeddings (this may take a minute first time)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Build the FAISS vector store from chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Document indexed successfully!")
    
    return vector_store

model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Friday, a helpful private AI assistant. 
    You have been given relevant excerpts from a document to help answer questions.
    Use the following context to answer the question accurately.
    If the answer is not in the context, say so honestly.
    
    Context from document:
    {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | model
chat_history = []


def chat_with_document(vector_store, question):
    relevant_chunks = vector_store.similarity_search(question, k=3)
    context = "\n \n".join([chunk.page_content for chunk in relevant_chunks])
    
    response = chain.invoke({
        "question": question,
        "chat_history": chat_history,
        "context": context
    })
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return response

if __name__ == "__main__":
    pdf_path = input("Enter path to your PDF file: ")
    vector_store = load_and_index_pdf(pdf_path)
    
    print("\n Friday is ready! Ask anything about your document.")
    print("Type 'quit' to exit")
    print("=" * 40)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = chat_with_document(vector_store, user_input)
        print(f"Friday: {response}\n")