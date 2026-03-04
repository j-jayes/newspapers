"""Minimal OpenAI-compatible VLM server for HuggingFace Inference Endpoints.

Serves an OCR vision-language model (GLM-OCR, DeepSeek-OCR-2, LightOnOCR-2, etc.)
via a ``/v1/chat/completions`` endpoint that accepts images as base64 data URIs.

The model is loaded from ``/repository`` (mounted by HF Inference Endpoints).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-server")

MODEL_PATH = os.environ.get("MODEL_PATH", "/repository")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))

# Globals populated at startup
model = None
processor = None
device = None
dtype = None


def load_model():
    """Auto-detect model architecture and load with the appropriate class."""
    global model, processor, device, dtype
    import json
    from pathlib import Path

    config_path = Path(MODEL_PATH) / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    model_type = config.get("model_type", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info("Model type: %s | Device: %s | Dtype: %s", model_type, device, dtype)

    if model_type == "deepseek_ocr_2":
        # DeepSeek-OCR-2: uses custom AutoModel with trust_remote_code
        from transformers import AutoModel, AutoTokenizer
        processor = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True, use_safetensors=True
        ).eval().to(device=device, dtype=dtype)
    elif model_type == "lightonocr":
        # LightOnOCR-2: has its own model class in transformers
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
        processor = LightOnOcrProcessor.from_pretrained(MODEL_PATH)
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=dtype
        ).to(device)
    else:
        # GLM-OCR and other standard VLMs
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

    logger.info("Model loaded successfully from %s", MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="OCR VLM Server", lifespan=lifespan)


# ── Pydantic models for OpenAI-compatible chat API ──────────────────────


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


# ── Helpers ──────────────────────────────────────────────────────────────


def _decode_image(data_uri: str) -> Image.Image:
    """Decode a base64 data URI or URL to a PIL Image."""
    if data_uri.startswith("data:"):
        # data:image/png;base64,<base64>
        header, b64data = data_uri.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    raise ValueError(f"Unsupported image source (only base64 data URIs supported): {data_uri[:60]}")


def _extract_from_request(req: ChatRequest) -> tuple[list[Image.Image], str]:
    """Extract images and text prompt from the chat request."""
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


def _run_inference(images: list[Image.Image], text: str, max_tokens: int) -> str:
    """Run model inference on images + text and return generated string."""
    import json
    from pathlib import Path

    config_path = Path(MODEL_PATH) / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    model_type = config.get("model_type", "")

    if model_type == "deepseek_ocr_2":
        # DeepSeek-OCR-2 uses model.infer() with its own tokenizer
        # For simplicity, save image to temp file and use infer()
        import tempfile
        if images:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            images[0].save(tmp.name)
            prompt = f"<image>\n{text}" if text else "<image>\nConvert the document to markdown."
            result = model.infer(
                processor, prompt=prompt, image_file=tmp.name,
                base_size=1024, image_size=768
            )
            os.unlink(tmp.name)
            return result if isinstance(result, str) else str(result)
        return ""

    elif model_type == "lightonocr":
        # LightOnOCR-2 uses chat template
        conversation = [{"role": "user", "content": []}]
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            conversation[0]["content"].append({"type": "image", "image": img})
        if text:
            conversation[0]["content"].append({"type": "text", "text": text})

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        return processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    else:
        # GLM-OCR and other standard AutoModelForImageTextToText
        messages = [{"role": "user", "content": []}]
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            messages[0]["content"].append({"type": "image", "url": f"data:image/png;base64,{b64}"})
        if text:
            messages[0]["content"].append({"type": "text", "text": text})

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        return processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


# ── Endpoints ────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> ChatResponse:
    images, text = _extract_from_request(req)
    logger.info("Chat request: %d images, %d chars text", len(images), len(text))
    t0 = time.time()
    result = _run_inference(images, text, req.max_tokens)
    elapsed = time.time() - t0
    logger.info("Inference done in %.1fs (%d chars output)", elapsed, len(result))
    return ChatResponse(choices=[Choice(message={"role": "assistant", "content": result})])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
