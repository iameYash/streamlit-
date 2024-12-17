import io
import json
import boto3
from botocore.config import Config
my_config = Config(
    region_name = 'us-west-1',
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)
import streamlit as st
from PIL import Image
st.title("Building with Bedrock")  # Title of the application
st.subheader("Model Playground")
# List of Stable Diffusion Preset Styles
sd_presets = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]
# Turn base64 string to image with PIL
def base64_to_pil(base64_string):
    """
    Purpose:
        Turn base64 string to image with PIL
    Args/Requests:
         base64_string: base64 string of image
    Return:
        image: PIL image
    """
    import base64
    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    return image
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    endpoint_url='https://vpce-071e0069bc4391e18.bedrock-runtime.us-west-1.vpce.amazonaws.com',
    config=my_config
)
# Bedrock api call to stable diffusion
def generate_image_sd(text, style):
    """
    Purpose:
        Uses Bedrock API to generate an Image
    Args/Requests:
         text: Prompt
         style: style for image
    Return:
        image: base64 string of image
    """
    body = {
        "text_prompts": [{"text": text}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": style,
    }
    if style == "None":
        del body["style_preset"]
    body = json.dumps(body)
    modelId = "stability.stable-diffusion-xl-v1"
    accept = "application/json"
    contentType = "application/json"
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    results = response_body.get("artifacts")[0].get("base64")
    return results
def call_claude_3(
    system_prompt: str,
    prompt: str,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
):
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0.7,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    body = json.dumps(prompt_config)
    modelId = model_id
    accept = "application/json"
    contentType = "application/json"
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    results = response_body.get("content")[0].get("text")
    return results
models = ["Stable Diffusion", "Claude 3 Haiku"]
current_model = st.selectbox("Select Model", models)
if current_model == "Stable Diffusion":
    # select box for styles
    style = st.selectbox("Select Style", sd_presets)
    # text input
    prompt = st.text_area("Enter prompt")
    #  Generate image from prompt,
    if st.button("Generate Image"):
        image_b64 = generate_image_sd(prompt, style)
        image = base64_to_pil(image_b64)
        st.image(image)
if current_model == "Claude 3 Haiku":
    system_prompt = st.text_area("Enter system prompt")
    prompt = st.text_area("Enter prompt")
    #  Generate text from prompt,
    if st.button("Call Claude"):
        generated_text = call_claude_3(system_prompt, prompt)
        st.markdown(generated_text)
