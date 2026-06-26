import os
import time

import gradio as gr
import requests

ENDPOINT_URL = os.getenv(
    "CLEARML_SERVING_URL", "http://127.0.0.1:8020/serve/class"
)

LABEL_MAP = {"1": "очень позитивно, круто", "0": "очень негативно, вообще не круто"}


def predict(text):
    if not text.strip():
        return "Введите текст", ""

    try:
        start = time.perf_counter()
        r = requests.post(
            ENDPOINT_URL,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        client_latency_ms = round((time.perf_counter() - start) * 1000, 2)

        r.raise_for_status()
        data = r.json()

        raw_label = data.get("label")
        label_text = LABEL_MAP.get(
            raw_label, LABEL_MAP.get(str(raw_label), str(raw_label))
        )

        return label_text, f"{client_latency_ms} ms"

    except requests.exceptions.RequestException as e:
        return "Ошибка", f"Endpoint недоступен: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# Sentiment Classification")
    inp = gr.Textbox(label="Текст", lines=4, placeholder="Введите предложение...")
    btn = gr.Button("Predict")
    label_out = gr.Textbox(label="Label")
    latency_out = gr.Textbox(label="Latency")
    btn.click(fn=predict, inputs=inp, outputs=[label_out, latency_out])

demo.launch(server_name="0.0.0.0", server_port=7860)
