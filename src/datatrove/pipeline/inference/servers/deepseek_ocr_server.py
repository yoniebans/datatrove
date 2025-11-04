import asyncio
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
from PIL import Image

from datatrove.pipeline.inference.servers import InferenceServer
from datatrove.utils._import_utils import check_required_dependencies
from loguru import logger


class DeepSeekOCRHandler(BaseHTTPRequestHandler):
    """HTTP handler for DeepSeek-OCR requests."""

    llm = None
    sampling_params = None

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            try:
                content_length = int(self.headers['Content-Length'])
                request_body = self.rfile.read(content_length)
                request_data = json.loads(request_body.decode('utf-8'))

                messages = request_data.get('messages', [])
                if not messages:
                    self.send_error(400, "No messages provided")
                    return

                image_data = None
                text_prompt = "Free OCR."

                for message in messages:
                    if message.get('role') == 'user':
                        content = message.get('content', [])
                        if isinstance(content, list):
                            for item in content:
                                if item.get('type') == 'image_url':
                                    image_url = item.get('image_url', {}).get('url', '')
                                    if image_url.startswith('data:image'):
                                        base64_data = image_url.split(',', 1)[1]
                                        image_bytes = base64.b64decode(base64_data)
                                        image_data = Image.open(io.BytesIO(image_bytes))
                                elif item.get('type') == 'text':
                                    text_content = item.get('text', '')
                                    if text_content and text_content != "Free OCR.":
                                        text_prompt = text_content

                if image_data is None:
                    self.send_error(400, "No image provided in request")
                    return

                prompt = f"<image>\n{text_prompt}"
                model_input = [{
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data}
                }]

                outputs = self.llm.generate(model_input, self.sampling_params)
                text = outputs[0].outputs[0].text

                prompt_tokens = len(prompt) // 4
                completion_tokens = len(text) // 4

                response_data = {
                    "choices": [{
                        "message": {
                            "content": text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }

                response_body = json.dumps(response_data).encode('utf-8')

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                self.send_error(500, f"Internal server error: {str(e)}")
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            response_data = {
                "object": "list",
                "data": [{"id": "deepseek-ai/DeepSeek-OCR", "object": "model"}]
            }
            response_body = json.dumps(response_data).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class DeepSeekOCRServer(InferenceServer):
    """DeepSeek-OCR inference server using vLLM Python API."""

    def __init__(self, model_name_or_path: str, max_context: int, model_kwargs: dict | None = None):
        check_required_dependencies("DeepSeek OCR server", [
            ("fitz", "pymupdf"),
            ("PIL", "pillow"),
            "vllm"
        ])
        super().__init__(model_name_or_path, max_context, model_kwargs)
        self.server: HTTPServer = None

    async def start_server_task(self) -> None:
        """Start the DeepSeek-OCR server with vLLM Python API."""
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        logger.info(f"Loading DeepSeek-OCR model: {self.model_name_or_path}")

        try:
            DeepSeekOCRHandler.llm = LLM(
                model=self.model_name_or_path,
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor],
                trust_remote_code=True,
            )
            logger.info("DeepSeek-OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DeepSeek-OCR model: {e}")
            raise

        DeepSeekOCRHandler.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_context,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        )

        def run_server():
            self.server = HTTPServer(('localhost', self.port), DeepSeekOCRHandler)
            logger.info(f"DeepSeek-OCR server started on port {self.port}")
            self.server.serve_forever()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        await asyncio.sleep(2)
        logger.info("DeepSeek-OCR server is ready")

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            if self.server:
                self.server.shutdown()
            raise

    async def is_ready(self) -> bool:
        """Check if DeepSeek-OCR server is ready."""
        url = f"http://localhost:{self.port}/v1/models"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url, timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False
