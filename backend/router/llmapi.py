from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional, List
import time
import json

from services.llm_services import LLMServices

router = APIRouter(prefix="/api", tags=["Chat"])


class ChatRequest(BaseModel):
    query: str
    context: Optional[List[str]] = None
    model: Optional[str] = None  # e.g. "qwen2.5:3b" - uses DEFAULT_LLM_PROVIDER if not qualified

    @field_validator("query")
    @classmethod
    def check_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()

    @field_validator("model")
    @classmethod
    def check_model(cls, v):
        if v is not None and not v.strip():
            raise ValueError("model cannot be empty string")
        return v.strip() if v else None


@router.post('/chat/stream')
async def chat(request: ChatRequest):

    try:
        message = {'role': 'user', 'content': request.query}

        # Per-request model override — functional picker for NiceGUI
        llm = LLMServices(model=request.model) if request.model else LLMServices()

        start_time = time.perf_counter()
        stream = llm.astream(messages=[message])

        # 1. Await the first delta before returning StreamingResponse to calculate TTFT
        #    Service now yields str (not litellm chunk), so TTFT still measured here via first await
        try:
            first_delta = await anext(stream)
            ttft_ms = (time.perf_counter() - start_time) * 1000
            print(f"[METRIC] Time To First Token (TTFT): {ttft_ms:.2f} ms")
        except StopAsyncIteration:
            first_delta = None
            ttft_ms = 0.0
        except Exception as e:
            return JSONResponse(
                status_code=500, content={"Exception occured": str(e), "type": type(e).__name__}
            )

        # 2. Generator that emits the cached first token, then continues the stream
        async def sse_wrap():
            if first_delta:
                yield f"data: {json.dumps({'content': first_delta})}\n\n"

            async for delta in stream:
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"

            yield "data: [DONE]\n\n"

        # 3. Pass the calculated TTFT to headers
        return StreamingResponse(
            sse_wrap(),
            media_type="text/event-stream",
            headers={
                "X-TTFT": f"{ttft_ms:.2f}",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"Exception occured": str(e), "type": type(e).__name__}
        )