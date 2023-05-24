import os

import gradio as gr
from huggingface_hub import Repository
from text_generation import Client

# from dialogues import DialogueTemplate
from share_btn import (community_icon_html, loading_icon_html, share_btn_css,
                       share_js)

HF_TOKEN = os.environ.get("HF_TOKEN", None)
API_TOKEN = os.environ.get("API_TOKEN", None)
API_URL = os.environ.get("API_URL", None)
API_URL = "https://api-inference.huggingface.co/models/timdettmers/guanaco-33b-merged"

client = Client(
    API_URL,
    headers={"Authorization": f"Bearer {API_TOKEN}"},
)

repo = None


def get_total_inputs(inputs, chatbot, preprompt, user_name, assistant_name, sep):
    past = []
    for data in chatbot:
        user_data, model_data = data

        if not user_data.startswith(user_name):
            user_data = user_name + user_data
        if not model_data.startswith(sep + assistant_name):
            model_data = sep + assistant_name + model_data

        past.append(user_data + model_data.rstrip() + sep)

    if not inputs.startswith(user_name):
        inputs = user_name + inputs

    total_inputs = preprompt + "".join(past) + inputs + sep + assistant_name.rstrip()

    return total_inputs


def has_no_history(chatbot, history):
    return not chatbot and not history


header = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
prompt_template = "### Human: {query}\n ### Assistant:{response}"

def generate(
    user_message,
    chatbot,
    history,
    temperature,
    top_p,
    max_new_tokens,
    repetition_penalty,
):
    # Don't return meaningless message when the input is empty
    if not user_message:
        print("Empty input")

    history.append(user_message)

    past_messages = []
    for data in chatbot:
        user_data, model_data = data

        past_messages.extend(
            [{"role": "user", "content": user_data}, {"role": "assistant", "content": model_data.rstrip()}]
        )
        
    if len(past_messages) < 1:
        prompt = header + prompt_template.format(query=user_message, response="")
    else:
        prompt = header
        for i in range(0, len(past_messages), 2):
            intermediate_prompt = prompt_template.format(query=past_messages[i]["content"], response=past_messages[i+1]["content"])
            print("intermediate: ", intermediate_prompt)
            prompt = prompt + '\n' + intermediate_prompt

        prompt = prompt + prompt_template.format(query=user_message, response="")


    generate_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        truncate=999,
        seed=42,
    )

    stream = client.generate_stream(
        prompt,
        **generate_kwargs,
    )

    output = ""
    for idx, response in enumerate(stream):
        if response.token.text == '':
            break

        if response.token.special:
            continue
        output += response.token.text
        if idx == 0:
            history.append(" " + output)
        else:
            history[-1] = output

        chat = [(history[i].strip(), history[i + 1].strip()) for i in range(0, len(history) - 1, 2)]

        yield chat, history, user_message, ""

    return chat, history, user_message, ""


examples = [
    "A Llama entered in my garden, what should I do?"
]


def clear_chat():
    return [], []


def process_example(args):
    for [x, y] in generate(args):
        pass
    return [x, y]


title = """<h1 align="center">Guanaco Playground 💬</h1>"""
custom_css = """
#banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""

with gr.Blocks(analytics_enabled=False, css=custom_css) as demo:
    gr.HTML(title)

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
            💻 This demo showcases the Guanaco 33B model, released together with the paper [QLoRA](https://arxiv.org/abs/2305.14314)
    """
            )

    with gr.Row():
        with gr.Box():
            output = gr.Markdown()
            chatbot = gr.Chatbot(elem_id="chat-message", label="Chat")

    with gr.Row():
        with gr.Column(scale=3):
            user_message = gr.Textbox(placeholder="Enter your message here", show_label=False, elem_id="q-input")
            with gr.Row():
                send_button = gr.Button("Send", elem_id="send-btn", visible=True)

                clear_chat_button = gr.Button("Clear chat", elem_id="clear-btn", visible=True)

            with gr.Accordion(label="Parameters", open=False, elem_id="parameters-accordion"):
                temperature = gr.Slider(
                    label="Temperature",
                    value=0.7,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    interactive=True,
                    info="Higher values produce more diverse outputs",
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    value=0.9,
                    minimum=0.0,
                    maximum=1,
                    step=0.05,
                    interactive=True,
                    info="Higher values sample more low-probability tokens",
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    value=1024,
                    minimum=0,
                    maximum=1024,
                    step=4,
                    interactive=True,
                    info="The maximum numbers of new tokens",
                )
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    value=1.2,
                    minimum=0.0,
                    maximum=10,
                    step=0.1,
                    interactive=True,
                    info="The parameter for repetition penalty. 1.0 means no penalty.",
                )
            with gr.Row():
                gr.Examples(
                    examples=examples,
                    inputs=[user_message],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )

    history = gr.State([])
    last_user_message = gr.State("")

    user_message.submit(
        generate,
        inputs=[
            user_message,
            chatbot,
            history,
            temperature,
            top_p,
            max_new_tokens,
            repetition_penalty,
        ],
        outputs=[chatbot, history, last_user_message, user_message],
    )

    send_button.click(
        generate,
        inputs=[
            user_message,
            chatbot,
            history,
            temperature,
            top_p,
            max_new_tokens,
            repetition_penalty,
        ],
        outputs=[chatbot, history, last_user_message, user_message],
    )

    clear_chat_button.click(clear_chat, outputs=[chatbot, history])

demo.queue(concurrency_count=16).launch(debug=True)
