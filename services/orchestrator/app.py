# services/orchestrator/app.py - Fixed WebSocket and Streaming Issues
from fastapi import FastAPI, WebSocket, HTTPException, Query, File, UploadFile, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
import httpx
import json
import asyncio
from typing import Dict, Optional, List, AsyncGenerator
import redis
import uuid
from datetime import datetime
import logging
import os
import traceback
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS Orchestrator", version="2.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
LLM_SERVICE = os.getenv('LLM_SERVICE_URL', 'http://llm-service:8001')
RAG_SERVICE = os.getenv('RAG_SERVICE_URL', 'http://rag-service:8002')
VOICE_SERVICE = os.getenv('VOICE_SERVICE_URL', 'http://voice-service:8003')
VLLM_URL = os.getenv('VLLM_URL', 'http://vllm:8000')
VLLM_API_KEY = os.getenv('VLLM_API_KEY', 'jarvis-key-123')

# Redis connection with error handling
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=6379,
        decode_responses=True,
        socket_connect_timeout=5,
        retry_on_timeout=True
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
        self.message_count = 0
    
    async def process_message(self, message: str, use_rag: bool = True) -> AsyncGenerator[str, None]:
        """Process a message and yield response chunks"""
        self.message_count += 1
        context_docs = []
        
        # RAG search if enabled
        if use_rag:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{RAG_SERVICE}/search",
                        json={
                            "query": message,
                            "collection": "documents",
                            "top_k": 3,
                            "threshold": 0.5
                        }
                    )
                    if response.status_code == 200:
                        result = response.json()
                        context_docs = result.get("results", [])
                        logger.info(f"RAG search returned {len(context_docs)} results")
            except httpx.ConnectError:
                logger.warning("RAG service unavailable - proceeding without context")
            except Exception as e:
                logger.error(f"RAG error: {e}")
        
        # Build prompt with context
        prompt = self._build_prompt(message, context_docs)
        
        # Stream response from LLM
        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{LLM_SERVICE}/generate",
                    json={
                        "prompt": prompt,
                        "session_id": self.session_id,
                        "stream": True,
                        "temperature": 0.7,
                        "max_tokens": 2048
                    }
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            full_response += line
                            yield line
                            
        except httpx.ConnectError:
            error_msg = "Error: Cannot connect to LLM service. Please ensure all services are running."
            logger.error(error_msg)
            yield error_msg
        except httpx.TimeoutException:
            error_msg = "Error: Request timed out. The model might be loading or overwhelmed."
            logger.error(error_msg)
            yield error_msg
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            yield f"Error: {str(e)}"
        
        # Store conversation in Redis
        if redis_client and full_response:
            try:
                conversation_data = {
                    "timestamp": datetime.now().isoformat(),
                    "user": message,
                    "assistant": full_response,
                    "context_used": len(context_docs) > 0,
                    "session_id": self.session_id
                }
                
                # Store in Redis
                key = f"conversation:{self.session_id}:{self.message_count}"
                redis_client.setex(key, 3600, json.dumps(conversation_data))
                
                # Update session info
                session_key = f"session:{self.session_id}"
                session_data = {
                    "last_activity": datetime.now().isoformat(),
                    "message_count": self.message_count,
                    "last_message": message[:100]
                }
                redis_client.setex(session_key, 3600, json.dumps(session_data))
                
            except Exception as e:
                logger.error(f"Redis storage error: {e}")
    
    def _build_prompt(self, message: str, context_docs: List[Dict]) -> str:
        """Build prompt with RAG context"""
        prompt_parts = []
        
        if context_docs:
            prompt_parts.append("Based on the following relevant information:")
            prompt_parts.append("")
            
            for i, doc in enumerate(context_docs[:3], 1):
                text = doc.get('text', '')[:500]
                source = doc.get('source', 'unknown')
                score = doc.get('score', 0)
                prompt_parts.append(f"[{i}] From {source} (relevance: {score:.2f}):")
                prompt_parts.append(text)
                prompt_parts.append("")
            
            prompt_parts.append("---")
            prompt_parts.append("")
        
        prompt_parts.append(f"User question: {message}")
        prompt_parts.append("")
        prompt_parts.append("Please provide a helpful and accurate response.")
        
        return "\n".join(prompt_parts)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, SessionManager] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.sessions[session_id] = SessionManager(session_id)
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.sessions:
            del self.sessions[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)
    
    async def send_text(self, session_id: str, text: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(text)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, session_id)
    
    try:
        # Send initial connection message
        await manager.send_message(session_id, {
            "type": "connection",
            "session_id": session_id,
            "message": "Connected to JARVIS. How can I assist you?"
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=300.0  # 5 minute timeout
                )
                
                # Handle different message types
                if data.get("type") == "ping":
                    await manager.send_message(session_id, {"type": "pong"})
                    continue
                
                elif data.get("type") == "text":
                    message = data.get("message", "")
                    use_rag = data.get("use_rag", True)
                    
                    if not message:
                        await manager.send_message(session_id, {
                            "type": "error",
                            "message": "Empty message received"
                        })
                        continue
                    
                    logger.info(f"Processing message from {session_id}: {message[:50]}...")
                    
                    # Get session manager
                    session_manager = manager.sessions.get(session_id)
                    if not session_manager:
                        session_manager = SessionManager(session_id)
                        manager.sessions[session_id] = session_manager
                    
                    # Stream response
                    try:
                        async for chunk in session_manager.process_message(message, use_rag):
                            await manager.send_text(session_id, chunk)
                        
                        # Send completion signal
                        await manager.send_message(session_id, {"type": "complete"})
                        
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        await manager.send_message(session_id, {
                            "type": "error",
                            "message": f"Error processing request: {str(e)}"
                        })
                
                elif data.get("type") == "voice":
                    # Handle voice input
                    try:
                        audio_data = data.get("audio")
                        
                        # Send to voice service for transcription
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.post(
                                f"{VOICE_SERVICE}/transcribe",
                                files={"audio": audio_data}
                            )
                            transcription = response.json()["text"]
                        
                        await manager.send_message(session_id, {
                            "type": "transcription",
                            "text": transcription
                        })
                        
                        # Process the transcribed text
                        session_manager = manager.sessions.get(session_id)
                        if not session_manager:
                            session_manager = SessionManager(session_id)
                            manager.sessions[session_id] = session_manager
                        
                        async for chunk in session_manager.process_message(transcription):
                            await manager.send_text(session_id, chunk)
                        
                        await manager.send_message(session_id, {"type": "complete"})
                        
                    except Exception as e:
                        logger.error(f"Voice processing error: {e}")
                        await manager.send_message(session_id, {
                            "type": "error",
                            "message": f"Voice processing error: {str(e)}"
                        })
                
            except asyncio.TimeoutError:
                logger.info(f"WebSocket timeout for session {session_id}")
                await manager.send_message(session_id, {
                    "type": "timeout",
                    "message": "Connection timeout due to inactivity"
                })
                break
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {session_id}")
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": "Invalid message format"
                })
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                logger.error(traceback.format_exc())
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Server error: {str(e)}"
                })
                break
    
    except Exception as e:
        logger.error(f"WebSocket fatal error: {traceback.format_exc()}")
    finally:
        manager.disconnect(session_id)

@app.post("/chat")
async def chat_endpoint(
    message: str = Query(..., description="User message"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    use_rag: bool = Query(True, description="Use RAG for context")
):
    """REST API endpoint for chat"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Get or create session
    if session_id not in manager.sessions:
        manager.sessions[session_id] = SessionManager(session_id)
    
    session = manager.sessions[session_id]
    
    # Collect response
    response_chunks = []
    async for chunk in session.process_message(message, use_rag):
        response_chunks.append(chunk)
    
    response = "".join(response_chunks)
    
    return {
        "session_id": session_id,
        "response": response,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "services": {},
        "active_sessions": len(manager.sessions),
        "active_websockets": len(manager.active_connections)
    }
    
    # Check service health
    services = {
        "llm": f"{LLM_SERVICE}/health",
        "rag": f"{RAG_SERVICE}/health",
        "voice": f"{VOICE_SERVICE}/health"
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services.items():
            try:
                response = await client.get(url)
                health_status["services"][name] = response.json()
            except Exception as e:
                logger.error(f"Health check for {name} failed: {e}")
                health_status["services"][name] = {"status": "unhealthy", "error": str(e)}
                health_status["status"] = "degraded"
    
    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status["services"]["redis"] = {"status": "healthy"}
        else:
            health_status["services"]["redis"] = {"status": "not configured"}
    except Exception as e:
        health_status["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check vLLM
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{VLLM_URL}/health",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
            )
            health_status["services"]["vllm"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy"
            }
    except Exception as e:
        health_status["services"]["vllm"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "JARVIS Orchestrator",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "websocket": "/ws",
            "chat": "/chat",
            "health": "/health",
            "documents": "/api/documents/upload"
        },
        "active_sessions": len(manager.sessions)
    }

# ============= Document Management API =============

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload document to RAG service"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(f"{RAG_SERVICE}/ingest", files=files)
            response.raise_for_status()
            
            result = response.json()
            
            # Store document info in Redis
            if redis_client and result.get("status") == "success":
                doc_info = {
                    "id": result.get("document_id"),
                    "filename": file.filename,
                    "chunks": result.get("chunks_created", 0),
                    "uploaded": datetime.now().isoformat(),
                    "collection": result.get("collection", "documents")
                }
                
                redis_client.lpush("jarvis:documents", json.dumps(doc_info))
                redis_client.ltrim("jarvis:documents", 0, 99)
            
            return result
            
    except httpx.HTTPStatusError as e:
        logger.error(f"RAG service error during upload: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    collection: str = Query("documents", description="Collection name"),
    top_k: int = Query(5, description="Number of results"),
    threshold: float = Query(0.5, description="Similarity threshold")
):
    """Search documents using RAG service"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_SERVICE}/search",
                json={
                    "query": query,
                    "collection": collection,
                    "top_k": top_k,
                    "threshold": threshold
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/collections")
async def list_collections():
    """List available collections from RAG service"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RAG_SERVICE}/collections")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(httpx.ConnectError)
async def connection_error_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": "Service temporarily unavailable. Please try again."}
    )

@app.exception_handler(httpx.TimeoutException)
async def timeout_error_handler(request, exc):
    return JSONResponse(
        status_code=504,
        content={"detail": "Request timeout. The operation is taking longer than expected."}
    )
