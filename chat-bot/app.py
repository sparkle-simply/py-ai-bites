import gradio as gr
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# === CONFIGURATION ===
MODEL_NAME = "llama3:8b"          # change to phi3:mini, mistral:7b, etc.
SYSTEM_PROMPT = """You are a helpful, friendly assistant.
Answer concisely and clearly. If you don't know something, say so."""

# Load model
llm = OllamaLLM(model=MODEL_NAME, temperature=0.7)

# Simple prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

# Build chain
chain = prompt | llm

def chat_with_ai(message, history):
    """Gradio chatbot function"""
    response = chain.invoke({"question": message})
    return response

# === GRADIO INTERFACE ===
with gr.Blocks(title="AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Chatbot (runs completely offline)")

    chatbot = gr.ChatInterface(
        fn=chat_with_ai,
        chatbot=gr.Chatbot(height=500, label="Chat"),
        textbox=gr.Textbox(placeholder="Type your message here...", container=False),
        title="AI Chatbot",
        description="Ask anything — your conversations never leave your computer.",
        examples=["What is the capital of France?", "Tell me a joke about programming", "Explain quantum computing in simple terms"]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)