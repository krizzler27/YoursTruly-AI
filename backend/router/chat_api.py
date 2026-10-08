from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List
import asyncio
import time
import json
import uuid

from db.db_engine import get_db
from core.logging import get_logger, get_request_id, set_conversation_id
from services.llama_engine import LlamaEngine
from services.llm_service import LLMService
from services.chat_services import ChatServices
from services.grounding_service import gate_draft_stream, resolve_filenames, should_gate
from services.rollup_job import rollup_job
from schemas.api_schemas import ChatRequest, ConversationCreateRequest, ConversationResponse, MessageResponse, ConversationUpdateRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post('/chat/stream')
async def chat(request: ChatRequest, http_request: Request, db: Session = Depends(get_db)):

    try:
        engine = LlamaEngine.get_instance("chat")
        if engine.is_generating():
            return JSONResponse(
                status_code=429,
                content={"detail": "System Busy - model is generating. Try again."},
            )

        chat_service = ChatServices(db)
        try:
            conversation = chat_service.ensure_conversation(
                request.conversation_id, title=request.query
            )
        except ValueError as e:
            return JSONResponse(status_code=404, content={"detail": str(e)})

        set_conversation_id(conversation.id)
        history_prev = chat_service.get_history(conversation.id, limit=5)
        chat_service.add_message(conversation.id, "user", request.query)
        try:
            chat_service.maybe_remember(request.query)
        except Exception:
            logger.debug("maybe_remember skipped")

        llm = LLMService()

        if request.model and request.model.strip():
            engine.switch_model(request.model)

        start_time = time.perf_counter()  # TTFT covers agent dispatch
        outcome = await asyncio.to_thread(
            chat_service.run_agentic, request.query, history_prev, conversation.id
        )
        messages, route = outcome["messages"], outcome["route"]
        hits = outcome.get("hits") or []

        stream = llm.astream_chat(
            messages=messages,
            request=http_request,
            temperature=0.5 if route == "RAG" else 0.6,
            repeat_penalty=1.1,  # chat generation: loops/echoes cost more than paraphrase risk
        )

        if should_gate(route, hits):
            return await _gated_response(
                stream,
                query=request.query,
                hits=hits,
                conversation=conversation,
                chat_service=chat_service,
                db=db,
                start_time=start_time,
                route=route,
            )

        try:
            first_delta = await anext(stream)
            ttft_ms = (time.perf_counter() - start_time) * 1000
            logger.info("TTFT %.2f ms - route=%s", ttft_ms, route)
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
                content={"error": msg, "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"},
                headers={"X-Conversation-Id": str(conversation.id)},
            )

        acc_parts: list[str] = []
        if first_delta:
            acc_parts.append(first_delta)

        async def sse_wrap():
            if first_delta:
                yield f"data: {json.dumps({'content': first_delta})}\n\n"

            try:
                async for delta in stream:
                    if delta:
                        acc_parts.append(delta)
                        yield f"data: {json.dumps({'content': delta})}\n\n"
            except Exception as e:
                logger.error("Chat stream interrupted, conversation=%s", conversation.id, exc_info=True)
                yield f"data: {json.dumps({'error': str(e)[:200]})}\n\n"

            full_response = "".join(acc_parts).strip()
            if full_response:
                try:
                    chat_service.add_message(conversation.id, "assistant", full_response)
                except Exception:
                    logger.error("Assistant reply shown but not saved, conversation=%s", conversation.id, exc_info=True)
                    yield f"data: {json.dumps({'warning': 'not saved'})}\n\n"
                else:
                    try:
                        rollup_job.submit(db, conversation.id)
                    except Exception:
                        logger.debug("maybe_rollup failed, conversation=%s", conversation.id)

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
            status_code=500, content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"}
        )


async def _gated_response(stream, query: str, hits: list, conversation, chat_service, db, start_time: float, route: str):
    """Buffer a RAG-with-hits draft, grade it, then emit.

    Gating needs the full draft before the user sees any byte, so the
    first byte to the client (and X-TTFT) now includes generation plus
    grade time on gated turns. Pass or skip streams the buffered deltas
    in order; fail substitutes the refusal template and persists that.
    """
    try:
        names = resolve_filenames(db, hits)
        send_parts, send_text, verdict = await gate_draft_stream(
            stream, query=query, hits=hits, filenames=names
        )
    except Exception as e:
        msg = str(e)
        if "System Busy" in msg:
            return JSONResponse(
                status_code=429,
                content={"detail": msg},
                headers={"X-Conversation-Id": str(conversation.id)},
            )
        logger.error("Chat stream interrupted, conversation=%s", conversation.id, exc_info=True)
        ttft_ms = (time.perf_counter() - start_time) * 1000
        err = str(e)[:200]

        async def gated_error_wrap():
            yield f"data: {json.dumps({'error': err})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            gated_error_wrap(),
            media_type="text/event-stream",
            headers={
                "X-TTFT": f"{ttft_ms:.2f}",
                "X-Conversation-Id": str(conversation.id),
                "X-Route": route,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    ttft_ms = (time.perf_counter() - start_time) * 1000
    logger.info("TTFT %.2f ms - route=%s", ttft_ms, route)
    logger.info("grounding gate grounded=%s route=%s hits=%s", verdict, route, len(hits))

    async def gated_wrap():
        for part in send_parts:
            yield f"data: {json.dumps({'content': part})}\n\n"

        if send_text.strip():
            try:
                chat_service.add_message(conversation.id, "assistant", send_text.strip())
            except Exception:
                logger.error("Assistant reply shown but not saved, conversation=%s", conversation.id, exc_info=True)
                yield f"data: {json.dumps({'warning': 'not saved'})}\n\n"
            else:
                try:
                    rollup_job.submit(db, conversation.id)
                except Exception:
                    logger.debug("maybe_rollup failed, conversation=%s", conversation.id)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gated_wrap(),
        media_type="text/event-stream",
        headers={
            "X-TTFT": f"{ttft_ms:.2f}",
            "X-Conversation-Id": str(conversation.id),
            "X-Route": route,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@router.post('/conversations', response_model=ConversationResponse, status_code=201)
def create_conversation(req: ConversationCreateRequest, db: Session = Depends(get_db)):
    """Create an empty chat shell (e.g. so files can attach before the first message)."""
    try:
        return ChatServices(db).ensure_conversation(None, title=req.title)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"}
        )


@router.get('/conversations', response_model=List[ConversationResponse])
def list_conversations(db: Session = Depends(get_db)):
    try:
        llm = ChatServices(db)
        rows = llm.list_conversations(limit=50)
        return rows
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"})


@router.get('/conversations/{conversation_id}/messages', response_model=List[MessageResponse])
def list_messages(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    try:
        llm = ChatServices(db)
        rows = llm.list_messages(conversation_id, limit=100)
        return rows
    except ValueError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"})


@router.patch('/conversations/{conversation_id}', response_model=ConversationResponse)
def update_conversation(conversation_id: uuid.UUID, req: ConversationUpdateRequest, db: Session = Depends(get_db)):
    try:
        svc = ChatServices(db)
        conv = None
        if req.title is not None:
            conv = svc.rename_conversation(conversation_id, req.title)
        if "tag" in req.model_fields_set:
            conv = svc.set_tag(conversation_id, req.tag)
        elif "topic" in req.model_fields_set:
            conv = svc.set_tag(conversation_id, req.topic)
        if "project_id" in req.model_fields_set:
            conv = svc.set_project(conversation_id, req.project_id)
        if conv is None:
            conv = svc.chat_repo.get_by_id(conversation_id)
            if conv is None:
                raise ValueError(f"Conversation {conversation_id} not found")
        return conv
    except ValueError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"})


@router.delete('/conversations/{conversation_id}')
def delete_conversation(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    try:
        llm = ChatServices(db)
        llm.delete_conversation(conversation_id)
        return JSONResponse(content={"status": "deleted", "id": str(conversation_id)})
    except ValueError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"})