from typing import Any


class Preprocess(object):
    def __init__(self):
        pass

    def preprocess(
        self, body: dict, state: dict, collect_custom_statistics_fn=None
    ) -> Any:
        text = body.get("text")
        if text is None:
            raise ValueError("Missing 'text' field in request body")
        return [text]

    def postprocess(
        self, data: Any, state: dict, collect_custom_statistics_fn=None
    ) -> dict:
        label = int(data[0]) if hasattr(data, "__len__") else int(data)
        return {"label": str(label)}
