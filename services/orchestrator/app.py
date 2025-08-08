from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import asyncio
from typing import Dict, Optional, List
import redis
import uuid
from datetime import datetime
import logging
import os
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS Orchestrator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_SERVICE = os.getenv('LLM_SERVICE_URL', 'http://llm-service:8001')
RAG_SERVICE = os.getenv('RAG_SERVICE_URL', 'http://rag-service:8002')
VOICE_SERVICE = os.getenv('VOICE_SERVICE_URL', 'http://voice-service:8003')

try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=6379,
        decode_responses=True,
        socket_connect_timeout=5
    )
    redis_client.ping()
    logger.info("Connected to Redis successfully")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

class SessionManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.context = []
    
    async def process_message(self, message: str, use_rag: bool = True):
        context_docs = []
        if use_rag:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{RAG_SERVICE}/search",
                        json={"query": message, "top_k": 3}
                    )
                    response.raise_for_status()
                    context_docs = response.json()["results"]
            except httpx.HTTPStatusError as e:
                logger.error(f"RAG service error: {e.response.text}")
                context_docs = []
            except Exception as e:
                logger.error(f"RAG service connection error: {e}")
                context_docs = []
        
        prompt = self._build_prompt(message, context_docs)
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LLM_SERVICE}/generate",
                json={
                    "prompt": prompt,
                    "session_id": self.session_id,
                    "stream": True
                }
            )
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line:
                    yield line
        
        self.context.append({
            "timestamp": datetime.now().isoformat(),
            "user": message,
            "context_used": len(context_docs) > 0
        })
        
        if redis_client:
            try:
                redis_client.setex(
                    f"session:{self.session_id}:context",
                    3600,
                    json.dumps(self.context[-10:])
                )
            except Exception as e:
                logger.error(f"Redis setex error: {e}")
    
    def _build_prompt(self, message: str, context_docs: List[Dict]) -> str:
        prompt_parts = []
        
        if context_docs:
            prompt_parts.append("Relevant information from knowledge base:")
            for doc in context_docs[:3]:
                prompt_parts.append(f"- {doc['text'][:500]}")
            prompt_parts.append("")
        
        prompt_parts.append(f"User message: {message}")
        
        return "\n".join(prompt_parts)

sessions: Dict[str, SessionManager] = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    session = SessionManager(session_id)
    sessions[session_id] = session
    
    try:
        await websocket.send_json({
            "type": "connection",
            "session_id": session_id,
            "message": "JARVIS online. How can I assist you?"
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "text":
                async for chunk in session.process_message(
                    data["message"],
                    use_rag=data.get("use_rag", True)
                ):
                    await websocket.send_text(chunk)
                
                await websocket.send_json({"type": "complete"})
            
            elif data["type"] == "voice":
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{VOICE_SERVICE}/transcribe",
                        files={"audio": data["audio"]}
                    )
                    transcription = response.json()["text"]
                
                await websocket.send_json({
                    "type": "transcription",
                    "text": transcription
                })
                
                async for chunk in session.process_message(transcription):
                    await websocket.send_text(chunk)
                
                await websocket.send_json({"type": "complete"})
    
    except Exception as e:
        logger.error(f"WebSocket error: {traceback.format_exc()}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        if session_id in sessions:
            del sessions[session_id]
        await websocket.close()

@app.post("/chat")
async def chat_endpoint(message: str, session_id: Optional[str] = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = SessionManager(session_id)
    
    session = sessions[session_id]
    
    response = ""
    async for chunk in session.process_message(message):
        response += chunk
    
    return {
        "session_id": session_id,
        "response": response
    }

@app.get("/health")
async def health_check():
    health_status = {}
    
    services = {
        "llm": f"{LLM_SERVICE}/health",
        "rag": f"{RAG_SERVICE}/health",
        "voice": f"{VOICE_SERVICE}/health"
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services.items():
            try:
                response = await client.get(url)
                health_status[name] = response.json()
            except Exception as e:
                logger.error(f"Health check for {name} failed: {e}")
                health_status[name] = {"status": "unhealthy"}
    
    try:
        if redis_client:
            redis_client.ping()
            health_status["redis"] = {"status": "healthy"}
        else:
            health_status["redis"] = {"status": "unhealthy", "message": "Redis client is not configured"}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["redis"] = {"status": "unhealthy"}
    
    all_healthy = all(
        s.get("status") == "healthy" 
        for s in health_status.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": health_status,
        "active_sessions": len(sessions)
    }

@app.get("/")
async def root():
    return {
        "service": "JARVIS Orchestrator",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "websocket": "/ws",
            "chat": "/chat",
            "health": "/health"
        }
    }
