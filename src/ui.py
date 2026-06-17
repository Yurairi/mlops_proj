import time
import requests
import gradio as gr

API_URL = "http://127.0.0.1:8020/predict"

LABEL_MAP = {"1": "очень позитивно, круто", "0": "очень негативно, вообще не круто"}


def predict(text):
    if not text.strip():
        return "Введите текст", ""

    try:
        start = time.perf_counter()
        r = requests.post(API_URL, json={"text": text}, timeout=10)
        client_latency_ms = round((time.perf_counter() - start) * 1000, 2)

        r.raise_for_status()
        data = r.json()

        raw_label = data.get("label")
        label_text = LABEL_MAP.get(
            raw_label, LABEL_MAP.get(str(raw_label), str(raw_label))
        )

        server_latency = data.get("latency_ms")
        if server_latency is not None:
            latency_text = f"{client_latency_ms} ms (server: {server_latency} ms)"
        else:
            latency_text = f"{client_latency_ms} ms"

        return label_text, latency_text

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
