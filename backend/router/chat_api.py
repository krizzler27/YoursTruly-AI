from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List
import asyncio
import time
import json
import uuid

from db.db_engine import get_db
from services.llama_engine import LlamaEngine
from services.llm_service import LLMService
from services.chat_services import ChatServices
from schemas.api_schemas import ChatRequest, ConversationCreateRequest, ConversationResponse, MessageResponse, ConversationUpdateRequest

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post('/chat/stream')
async def chat(request: ChatRequest, http_request: Request, db: Session = Depends(get_db)):

    try:
        engine = LlamaEngine.get_instance()
        if engine.is_generating():
            return JSONResponse(
                status_code=429,
                content={"detail": "System Busy — model is generating. Try again."},
            )

        chat_service = ChatServices(db)
        try:
            conversation = chat_service.ensure_conversation(
                request.conversation_id, title=request.query
            )
        except ValueError as e:
            return JSONResponse(status_code=404, content={"detail": str(e)})

        history_prev = chat_service.get_history(conversation.id, limit=5)
        chat_service.add_message(conversation.id, "user", request.query)

        llm = LLMService()

        if request.model and request.model.strip():
            engine.switch_model(request.model)

        start_time = time.perf_counter()  # TTFT covers agent dispatch
        outcome = await asyncio.to_thread(
            chat_service.run_agentic, request.query, history_prev, conversation.id
        )
        messages, route = outcome["messages"], outcome["route"]

        stream = llm.astream_chat(
            messages=messages,
            request=http_request,
            temperature=0.5 if route == "RAG" else 0.6,
            repeat_penalty=1.1,  # chat generation: loops/echoes cost more than paraphrase risk
        )

        try:
            first_delta = await anext(stream)
            ttft_ms = (time.perf_counter() - start_time) * 1000
            print(f"[METRIC] Time To First Token (TTFT): {ttft_ms:.2f} ms")
        except StopAsyncIteration:
            first_delta = None
            ttft_ms = 0.0
        except Exception as e:
            msg = str(e)
            if "System Busy" in msg:
                return JSONResponse(
                    status_code=429,
                    content={"detail": msg},
                    headers={"X-Conversation-Id": str(conversation.id)},
                )
            return JSONResponse(
                status_code=500,
                content={"Exception occured": msg, "type": type(e).__name__},
                headers={"X-Conversation-Id": str(conversation.id)},
            )

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

            full_response = "".join(acc_parts).strip()
            if full_response:
                try:
                    chat_service.add_message(conversation.id, "assistant", full_response)
                except Exception:
                    pass

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_wrap(),
            media_type="text/event-stream",
            headers={
                "X-TTFT": f"{ttft_ms:.2f}",
                "X-Conversation-Id": str(conversation.id),
                "X-Route": route,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"Exception occured": str(e), "type": type(e).__name__}
        )


@router.post('/conversations', response_model=ConversationResponse, status_code=201)
def create_conversation(req: ConversationCreateRequest, db: Session = Depends(get_db)):
    """Create an empty chat shell (e.g. so files can attach before the first message)."""
    try:
        return ChatServices(db).ensure_conversation(None, title=req.title)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"Exception occured": str(e), "type": type(e).__name__}
        )


@router.get('/conversations', response_model=List[ConversationResponse])
def list_conversations(db: Session = Depends(get_db)):
    try:
        llm = ChatServices(db)
        rows = llm.list_conversations(limit=50)
        return rows
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.get('/conversations/{conversation_id}/messages', response_model=List[MessageResponse])
def list_messages(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    try:
        llm = ChatServices(db)
        rows = llm.list_messages(conversation_id, limit=100)
        return rows
    except ValueError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.patch('/conversations/{conversation_id}', response_model=ConversationResponse)
def rename_conversation(conversation_id: uuid.UUID, req: ConversationUpdateRequest, db: Session = Depends(get_db)):
    try:
        llm = ChatServices(db)
        conv = llm.rename_conversation(conversation_id, req.title)
        return conv
    except ValueError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.delete('/conversations/{conversation_id}')
def delete_conversation(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    try:
        llm = ChatServices(db)
        llm.delete_conversation(conversation_id)
        return JSONResponse(content={"status": "deleted", "id": str(conversation_id)})
    except ValueError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})