"""Minimal OpenAI-compatible server for DeepSeek-OCR-2.

Requires transformers==4.46.3 (custom code uses LlamaFlashAttention2).
Model is loaded from /repository (mounted by HF Inference Endpoints).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek-ocr")

MODEL_PATH = os.environ.get("MODEL_PATH", "/repository")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))

model = None
tokenizer = None
device = None


def load_model():
    global model, tokenizer, device
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info("Loading DeepSeek-OCR-2 from %s on %s", MODEL_PATH, device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True,
    ).eval().to(device=device, dtype=dtype)

    logger.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="DeepSeek-OCR-2 Server", lifespan=lifespan)


class ImageURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = MAX_NEW_TOKENS
    model: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str = "ocr-0"
    object: str = "chat.completion"
    choices: list[Choice]


def _decode_image(data_uri: str) -> Image.Image:
    if data_uri.startswith("data:"):
        header, b64data = data_uri.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    raise ValueError(f"Unsupported image source: {data_uri[:60]}")


def _extract_from_request(req: ChatRequest) -> tuple[list[Image.Image], str]:
    images: list[Image.Image] = []
    text_parts: list[str] = []
    for msg in req.messages:
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "text" and part.text:
                    text_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    images.append(_decode_image(part.image_url.url))
    return images, "\n".join(text_parts)


def _run_inference(images: list[Image.Image], text: str) -> str:
    if not images:
        return ""
    tmp_dir = tempfile.mkdtemp()
    tmp_img = os.path.join(tmp_dir, "input.png")
    images[0].save(tmp_img)
    prompt = f"<image>\n{text}" if text else "<image>\nConvert the document to markdown. "

    model.infer(
        tokenizer,
        prompt=prompt,
        image_file=tmp_img,
        output_path=tmp_dir,
        base_size=1024,
        image_size=768,
        crop_mode=True,
        save_results=True,
    )

    # Read the output file written by model.infer()
    result = ""
    result_file = os.path.join(tmp_dir, "result.mmd")
    if os.path.exists(result_file):
        with open(result_file) as f:
            result = f.read()

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> ChatResponse:
    images, text = _extract_from_request(req)
    logger.info("Request: %d images, %d chars text", len(images), len(text))
    t0 = time.time()
    result = _run_inference(images, text)
    elapsed = time.time() - t0
    logger.info("Inference done in %.1fs (%d chars)", elapsed, len(result))
    return ChatResponse(choices=[Choice(message={"role": "assistant", "content": result})])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
