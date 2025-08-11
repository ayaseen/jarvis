# services/orchestrator/app.py
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
import psycopg2
from psycopg2.extras import RealDictCursor

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

# Service URLs from environment
LLM_SERVICE = os.getenv('LLM_SERVICE_URL', 'http://llm-service:8001')
RAG_SERVICE = os.getenv('RAG_SERVICE_URL', 'http://rag-service:8002')
VOICE_SERVICE = os.getenv('VOICE_SERVICE_URL', 'http://voice-service:8003')
VLLM_URL = os.getenv('VLLM_URL', 'http://vllm:8000')
VLLM_API_KEY = os.getenv('VLLM_API_KEY', 'jarvis-key-123')
MODEL_NAME = os.getenv('MODEL_NAME', 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ')

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

# PostgreSQL connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            database=os.getenv('POSTGRES_DB', 'jarvis'),
            user=os.getenv('POSTGRES_USER', 'jarvis'),
            password=os.getenv('POSTGRES_PASSWORD', 'jarvis123'),
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

# Active WebSocket connections and session contexts
active_connections: Dict[str, WebSocket] = {}
session_contexts: Dict[str, List] = {}

@app.get("/")
async def root():
    return {
        "message": "JARVIS Orchestrator Service",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "websocket": "/ws",
            "chat": "/api/chat",
            "stream": "/api/chat/stream",
            "models": "/api/models",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    services_status = {}
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Check LLM service
        try:
            response = await client.get(f"{LLM_SERVICE}/health")
            services_status["llm"] = response.status_code == 200
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            services_status["llm"] = False
        
        # Check RAG service
        try:
            response = await client.get(f"{RAG_SERVICE}/health")
            services_status["rag"] = response.status_code == 200
        except Exception as e:
            logger.error(f"RAG health check failed: {e}")
            services_status["rag"] = False
        
        # Check Voice service
        try:
            response = await client.get(f"{VOICE_SERVICE}/health")
            services_status["voice"] = response.status_code == 200
        except Exception as e:
            logger.error(f"Voice health check failed: {e}")
            services_status["voice"] = False
        
        # Check vLLM
        try:
            response = await client.get(f"{VLLM_URL}/health")
            services_status["vllm"] = response.status_code == 200
        except Exception as e:
            logger.error(f"vLLM health check failed: {e}")
            services_status["vllm"] = False
    
    # Check Redis
    try:
        services_status["redis"] = redis_client.ping() if redis_client else False
    except:
        services_status["redis"] = False
    
    # Check PostgreSQL
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            services_status["postgres"] = True
        else:
            services_status["postgres"] = False
    except:
        services_status["postgres"] = False
    
    all_healthy = all(services_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services_status,
        "timestamp": datetime.utcnow().isoformat(),
        "active_connections": len(active_connections)
    }

@app.get("/api/models")
async def list_models():
    """List available models from vLLM"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{VLLM_URL}/v1/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"}  # FIXED: Added auth header
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"vLLM models endpoint returned {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to get models from vLLM: {e}")
    
    # Return default model if vLLM request fails
    return {
        "data": [
            {"id": MODEL_NAME, "object": "model", "owned_by": "jarvis"}
        ],
        "object": "list"
    }

@app.post("/api/chat")
async def chat_endpoint(request: dict):
    """Synchronous chat endpoint"""
    try:
        message = request.get("message", "")
        use_rag = request.get("use_rag", True)
        session_id = request.get("session_id", str(uuid.uuid4()))
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Get context from session
        context = session_contexts.get(session_id, [])
        
        # RAG search if enabled
        context_docs = []
        if use_rag:
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
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
                        logger.info(f"RAG found {len(context_docs)} documents")
                except Exception as e:
                    logger.error(f"RAG search failed: {e}")
        
        # Prepare prompt with context
        prompt = message
        if context_docs:
            context_text = "\n".join([doc.get("text", "") for doc in context_docs])
            prompt = f"Context:\n{context_text}\n\nQuestion: {message}"
        
        # Add conversation history
        if context:
            history = "\n".join([
                f"User: {turn['user']}\nAssistant: {turn['assistant']}" 
                for turn in context[-3:]  # Last 3 exchanges
            ])
            prompt = f"Previous conversation:\n{history}\n\nCurrent question: {prompt}"
        
        # Generate response using vLLM
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{VLLM_URL}/v1/completions",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "stream": False
                },
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["text"].strip()
                
                # Update session context
                context.append({"user": message, "assistant": answer})
                session_contexts[session_id] = context[-10:]  # Keep last 10 exchanges
                
                # Store in Redis if available
                if redis_client:
                    try:
                        redis_client.setex(
                            f"session:{session_id}",
                            3600,  # 1 hour expiry
                            json.dumps(context)
                        )
                    except Exception as e:
                        logger.error(f"Redis storage failed: {e}")
                
                return {
                    "response": answer,
                    "session_id": session_id,
                    "context_used": len(context_docs) > 0
                }
            else:
                error_msg = f"vLLM returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail="LLM generation failed")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: dict):
    """Streaming chat endpoint using SSE"""
    message = request.get("message", "")
    use_rag = request.get("use_rag", True)
    session_id = request.get("session_id", str(uuid.uuid4()))
    
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    async def generate():
        try:
            # RAG search
            context_docs = []
            if use_rag:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    try:
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
                            context_docs = response.json().get("results", [])
                    except Exception as e:
                        logger.error(f"RAG search failed: {e}")
            
            # Prepare prompt
            prompt = message
            if context_docs:
                context_text = "\n".join([doc.get("text", "") for doc in context_docs])
                prompt = f"Context:\n{context_text}\n\nQuestion: {message}"
            
            # Stream from vLLM
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{VLLM_URL}/v1/completions",
                    json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "max_tokens": 512,
                        "temperature": 0.7,
                        "stream": True
                    },
                    headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                yield f"data: [DONE]\n\n"
                            else:
                                yield f"data: {data}\n\n"
                    
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket
    logger.info(f"WebSocket connection established: {session_id}")
    
    try:
        # FIXED: Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "session_id": session_id,
            "message": "Connected to JARVIS. How can I assist you?"
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            use_rag = message_data.get("use_rag", True)
            
            if not user_message:
                continue
            
            # Send processing acknowledgment
            await websocket.send_json({
                "type": "processing",
                "message": "Processing your request..."
            })
            
            try:
                # RAG search
                context_docs = []
                if use_rag:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        try:
                            response = await client.post(
                                f"{RAG_SERVICE}/search",
                                json={
                                    "query": user_message,
                                    "collection": "documents",
                                    "top_k": 3,
                                    "threshold": 0.5
                                }
                            )
                            if response.status_code == 200:
                                context_docs = response.json().get("results", [])
                        except Exception as e:
                            logger.error(f"RAG search failed: {e}")
                
                # Prepare prompt
                prompt = user_message
                if context_docs:
                    context_text = "\n".join([doc.get("text", "") for doc in context_docs])
                    prompt = f"Context:\n{context_text}\n\nQuestion: {user_message}"
                
                # Stream response from vLLM
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream(
                        "POST",
                        f"{VLLM_URL}/v1/completions",
                        json={
                            "model": MODEL_NAME,
                            "prompt": prompt,
                            "max_tokens": 512,
                            "temperature": 0.7,
                            "stream": True
                        },
                        headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
                    ) as response:
                        full_response = ""
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    # Send completion message
                                    await websocket.send_json({
                                        "type": "complete",
                                        "message": full_response,
                                        "context_used": len(context_docs) > 0
                                    })
                                else:
                                    try:
                                        chunk = json.loads(data)
                                        text = chunk["choices"][0].get("text", "")
                                        if text:
                                            full_response += text
                                            # Send chunk to client
                                            await websocket.send_json({
                                                "type": "chunk",
                                                "content": text
                                            })
                                    except json.JSONDecodeError:
                                        pass
                        
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}\n{traceback.format_exc()}")
    finally:
        if session_id in active_connections:
            del active_connections[session_id]
        logger.info(f"WebSocket connection closed: {session_id}")

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload document for RAG indexing"""
    try:
        # Save file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Send to RAG service for indexing
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": (file.filename, f, file.content_type)}
                response = await client.post(
                    f"{RAG_SERVICE}/index",
                    files=files
                )
        
        # Clean up temporary file
        os.remove(file_path)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Document indexing failed: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text using voice service"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(
                f"{VOICE_SERVICE}/transcribe",
                files=files
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Transcription failed"
                )
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/synthesize")
async def synthesize_speech(request: dict):
    """Convert text to speech using voice service"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{VOICE_SERVICE}/synthesize",
                json={"text": text}
            )
            if response.status_code == 200:
                return Response(
                    content=response.content,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": "attachment; filename=speech.wav"
                    }
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Synthesis failed"
                )
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/collections")
async def list_collections():
    """List RAG collections"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RAG_SERVICE}/collections")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"Failed to get collections: {e}")
    
    return {"collections": []}

@app.delete("/api/cache/clear")
async def clear_cache():
    """Clear Redis cache"""
    try:
        if redis_client:
            redis_client.flushdb()
            return {"message": "Cache cleared successfully"}
        else:
            return {"message": "Redis not available"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)
