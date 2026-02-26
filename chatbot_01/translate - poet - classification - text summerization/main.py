import os
import gradio as gr

from app.use_cases.summarizer import Summarizer
from app.use_cases.classifier import TextClassifier
from app.use_cases.poem import Poet

from infra.llm.client import OpenAIClient
from infra.llm.ollama_client import OllamaClient

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "app/prompts")
prompt_files = [f for f in os.listdir(PROMPT_DIR) if f.endswith(".txt")]


def get_client(model_choice):
    if model_choice == "OpenAI":
        return OpenAIClient()
    return OllamaClient()


# -------- Summarization --------
def summarize_text(input_text, model_choice, prompt_choice):
    client = get_client(model_choice)
    summarizer = Summarizer(llm_client=client)

    prompt_path = os.path.join(PROMPT_DIR, prompt_choice)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(text_to_summarize=input_text)
    return summarizer.client.predict(prompt)


# -------- Classification --------
def classify_text(input_text, model_choice, prompt_choice):
    client = get_client(model_choice)
    classifier = TextClassifier(llm_client=client)

    prompt_path = os.path.join(PROMPT_DIR, prompt_choice)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(text=input_text)
    return classifier.client.predict(prompt)

# -------- Poet --------

def poet_generator(input_text, model_choice, prompt_choice):
    client = get_client(model_choice)
    classifier = TextClassifier(llm_client=client)

    prompt_path = os.path.join(PROMPT_DIR, prompt_choice)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(poet_text=input_text)
    return classifier.client.predict(prompt)

# -------- Translate --------

def translate(input_text, model_choice,language, prompt_choice):
    client = get_client(model_choice)
    classifier = TextClassifier(llm_client=client)

    prompt_path = os.path.join(PROMPT_DIR, prompt_choice)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()


    prompt = prompt_template.format(translate_text=input_text, language =language)
    return classifier.client.predict(prompt)

# -------- UI --------
with gr.Blocks(title="AI Text Engine") as demo:

    gr.Markdown("# 🧠 AI Text Engine\nThis agentic chatbot is for these tasks: Summarization / Classification / Poet Generation / Translator")

    with gr.Tabs():

        # -------- Tab 1: Summarization --------
        with gr.Tab("Summarization"):
            gr.Interface(
                fn=summarize_text,
                inputs=[
                    gr.Textbox(lines=10, label="Input Text"),
                    gr.Dropdown(["OpenAI", "Ollama"], value="OpenAI", label="Model"),
                    gr.Dropdown(
                        [p for p in prompt_files if p.startswith("summarization")],
                        label="Prompt"
                    ),
                ],
                outputs=gr.Textbox(lines=10, label="Summary"),
            )

        # -------- Tab 2: Classification --------
        with gr.Tab("Text Classification"):
            gr.Interface(
                fn=classify_text,
                inputs=[
                    gr.Textbox(lines=6, label="Input Text"),
                    gr.Dropdown(["OpenAI", "Ollama"], value="OpenAI", label="Model"),
                    gr.Dropdown(
                        [p for p in prompt_files if p.startswith("classification")],
                        label="Prompt"
                    ),
                ],
                outputs=gr.Textbox(label="Category"),
            )

        
        # -------- Tab 3: poet --------
        with gr.Tab("Poet Generator"):
            gr.Interface(
                fn=poet_generator,
                inputs=[
                    gr.Textbox(lines=6, label="Input Text"),
                    gr.Dropdown(["OpenAI", "Ollama"], value="OpenAI", label="Model"),
                    gr.Dropdown(
                        [p for p in prompt_files if p.startswith("poet")],
                        label="Prompt"
                    ),
                ],
                outputs=gr.Textbox(label="poet"),
            )
         # -------- Tab4: Translate --------
        with gr.Tab("Translator"):
            gr.Interface(
                fn=translate,
                inputs=[
                    gr.Textbox(lines=6, label="Input Text"),
                    gr.Dropdown(["OpenAI", "Ollama"], value="OpenAI", label="Model"),
                    gr.Dropdown(["Persian", "English", "chinese"] , label =" language to translate"),

                    gr.Dropdown(
                        [p for p in prompt_files if p.startswith("translate")],
                        label="Prompt"
                    ),
                ],
                outputs=gr.Textbox(label="translate"),
            )


if __name__ == "__main__":
    demo.launch()
