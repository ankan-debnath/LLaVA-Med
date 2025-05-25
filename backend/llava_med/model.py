# 1. Import your utility functions
from llava.mm_utils import process_images, tokenizer_image_token  # assuming your code is saved as utils.py
from .model_loader import get_model_components
from llava.conversation import conv_templates
import torch

from PIL import Image

tokenizer, model, image_processor, context_len = get_model_components()


# 2. Load and preprocess the image
def get_image_tensors(image_path):
    image = Image.open(image_path).convert("RGB")
    print(type(image))

    # 3. Preprocess using your utility (this uses padding, normalization, etc.)
    image_tensor = process_images([image], image_processor, model.config).to("cuda").half()
    return image_tensor

# 3. Create image tensors
def create_conversations(question):
    prompt = (
        "Analyze the following pathology case. "
        "The original image represents the full tissue, and the patches are selected regions of interest. "
        "Please describe abnormalities, patterns, or diagnoses suggested by these images."
    )
    question = prompt + question
    conv = conv_templates["v1_mmtag"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move tokenized input to CUDA individually
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    return input_ids, attention_mask


def generate_response(input_ids, attention_mask, image_tensor=None):
    with torch.no_grad():
        output_ids = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.95
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Answer:", output_text)

    return output_text
