from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
import time
import json

from db.db_engine import get_db
from services.llm_services import LLMServices
from services.chat_services import ChatServices
from schemas.api_schemas import ChatRequest

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post('/chat/stream')
async def chat(request: ChatRequest, db: Session = Depends(get_db)):

    try:
        chat_service = ChatServices(db)
        try:
            conversation = chat_service.ensure_conversation(
                request.conversation_id, title=request.query
            )
        except ValueError as e:
            return JSONResponse(status_code=404, content={"detail": str(e)})

        history_prev = chat_service.get_history(conversation.id, limit=5)
        chat_service.add_message(conversation.id, "user", request.query)

        llm = LLMServices(model=request.model) if request.model else LLMServices()
        messages = LLMServices.build_chat_messages(history_prev, request.query)

        start_time = time.perf_counter()
        stream = llm.astream(messages=messages)

        # 1. Await the first delta before returning StreamingResponse to calculate TTFT
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
        # accumulate for assistant persistence
        acc_parts: list[str] = []
        if first_delta:
            acc_parts.append(first_delta)

        async def sse_wrap():
            if first_delta:
                yield f"data: {json.dumps({'content': first_delta})}\n\n"

            async for delta in stream:
                if delta:
                    acc_parts.append(delta)
                    yield f"data: {json.dumps({'content': delta})}\n\n"

            # persist assistant message after stream completes
            full_response = "".join(acc_parts).strip()
            if full_response:
                try:
                    chat_service.add_message(conversation.id, "assistant", full_response)
                except Exception:
                    pass

            yield "data: [DONE]\n\n"

        # 3. Pass the calculated TTFT + conversation id to headers
        return StreamingResponse(
            sse_wrap(),
            media_type="text/event-stream",
            headers={
                "X-TTFT": f"{ttft_ms:.2f}",
                "X-Conversation-Id": str(conversation.id),
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"Exception occured": str(e), "type": type(e).__name__}
        )