import json
import os
import gradio as gr
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below:
Here is the conversation history: {context}
Question: {question}
Answer:
"""

model = OllamaLLM(model="granite3.1-dense")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def load_history():
    if os.path.exists("conversation_history.json"):
        with open("conversation_history.json", "r") as f:
            return json.load(f)
    return {}

def save_history(history):
    with open("conversation_history.json", "w") as f:
        json.dump(history, f, indent=4)

def chat(message, history):
    conversation_context = ""
    stored_history = load_history()
    
    # Build context from current session and stored history
    if history:
        for human, ai in history:
            conversation_context += f"\nUser: {human}\nAI: {ai}"
    
    # Check if question was asked before
    if message in stored_history:
        response = stored_history[message]
    else:
        response = chain.invoke({
            "context": conversation_context,
            "question": message
        })
        stored_history[message] = response
        save_history(stored_history)
    
    return response

# Create Gradio interface
bot = gr.ChatInterface(
    fn=chat,
    title="Granite 3.1 Chat",
    description="Chat with Granite 3.1 model. Your conversations are saved automatically.",
    examples=["Tell me about yourself", "What is machine learning?", "How does DNA work?"],
    cache_examples=False,
    stop_btn=True,
    run_examples_on_click=True,
    autoscroll=True,
)

if __name__ == "__main__":
    bot.launch()