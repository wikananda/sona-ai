from fastapi import APIRouter, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
import json

from sona_ai.api.routes._errors import route_error_handler
from sona_ai.api.schemas.chatbot import RecordingChatRequest
from sona_ai.db.models import Recording
from sona_ai.db.session import get_db
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select
from sona_ai.chatbot.openai_compatible_chat import OpenAICompatibleChat, _chat_context_from_segments

router = APIRouter()

@router.post("/recordings/{recording_id}/chat")
async def chat(
    recording_id: str,
    body: RecordingChatRequest,
    db: Session = Depends(get_db),
):
    recording = db.scalar(
        select(Recording)
        .where(Recording.id == recording_id)
        .options(selectinload(Recording.transcript))
    )

    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    if recording.transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    transcript = _chat_context_from_segments(
        json.loads(recording.transcript.segments_json)
    )
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty")
    
    byok = body.byok_settings
    
    with route_error_handler("chat error with recording: %s"):
        chat = OpenAICompatibleChat()
        answer = await run_in_threadpool(
            chat.answer,
            transcript,
            question,
            [message.model_dump() for message in body.history],
            byok.api_key,
            byok.model,
            byok.provider,
            byok.base_url,
        )

    return {"answer": answer}
