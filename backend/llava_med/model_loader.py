# model_loader.py

from llava.model.builder import load_pretrained_model

_model_cache = {}


def get_model_components():
    if "components" not in _model_cache:
        print("Loading LLaVA-Med model...")

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path='microsoft/llava-med-v1.5-mistral-7b',
            model_base=None,
            model_name='llava-med-v1.5-mistral-7b'
        )

        _model_cache["components"] = (tokenizer, model, image_processor, context_len)
    return _model_cache["components"]
