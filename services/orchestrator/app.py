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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS Orchestrator", version="1.0.0")

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

# Redis connection
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=6379,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
        socket_keepalive_options={
            1: 1,  # TCP_KEEPIDLE
            2: 3,  # TCP_KEEPINTVAL  
            3: 5   # TCP_KEEPCNT
        }
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
    
    async def process_message(self, message: str, use_rag: bool = True) -> AsyncGenerator[str, None]:
        context_docs = []
        
        # RAG search if enabled
        if use_rag:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{RAG_SERVICE}/search",
                        json={"query": message, "top_k": 3}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        context_docs = result.get("results", [])
                        logger.info(f"RAG search returned {len(context_docs)} results")
            except httpx.HTTPStatusError as e:
                logger.error(f"RAG service error: {e.response.text}")
            except httpx.ConnectError:
                logger.warning("RAG service unavailable")
            except Exception as e:
                logger.error(f"RAG error: {e}")
        
        # Build prompt with context
        prompt = self._build_prompt(message, context_docs)
        
        # Stream response from LLM
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
                            yield line
        except httpx.ConnectError:
            yield "Error: Cannot connect to LLM service. Please ensure all services are running."
        except httpx.TimeoutException:
            yield "Error: Request timed out. The model might be loading or overwhelmed."
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            yield f"Error: {str(e)}"
        
        # Store context for history
        self.context.append({
            "timestamp": datetime.now().isoformat(),
            "user": message,
            "context_used": len(context_docs) > 0
        })
        
        # Update Redis if available
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
                text = doc.get('text', '')[:500]
                source = doc.get('source', 'unknown')
                prompt_parts.append(f"- From {source}: {text}")
            prompt_parts.append("")
        
        prompt_parts.append(f"User message: {message}")
        
        return "\n".join(prompt_parts)

# Session storage
sessions: Dict[str, SessionManager] = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    session = SessionManager(session_id)
    sessions[session_id] = session
    
    logger.info(f"WebSocket connection established: {session_id}")
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "session_id": session_id,
            "message": "JARVIS online. How can I assist you?"
        })
        
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=300.0  # 5 minute timeout
                )
                
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                
                if data.get("type") == "text":
                    message = data.get("message", "")
                    use_rag = data.get("use_rag", True)
                    
                    if not message:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Empty message received"
                        })
                        continue
                    
                    logger.info(f"Processing message: {message[:50]}...")
                    
                    # Stream the response
                    try:
                        async for chunk in session.process_message(message, use_rag):
                            await websocket.send_text(chunk)
                        
                        # Send completion signal
                        await websocket.send_json({"type": "complete"})
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error processing request: {str(e)}"
                        })
                
                elif data.get("type") == "voice":
                    # Handle voice input
                    try:
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
                        
                        # Process the transcribed text
                        async for chunk in session.process_message(transcription):
                            await websocket.send_text(chunk)
                        
                        await websocket.send_json({"type": "complete"})
                    except Exception as e:
                        logger.error(f"Voice processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Voice processing error: {str(e)}"
                        })
                        
            except asyncio.TimeoutError:
                logger.info(f"WebSocket timeout for session {session_id}")
                await websocket.send_json({
                    "type": "timeout",
                    "message": "Connection timeout due to inactivity"
                })
                break
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {session_id}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid message format"
                })
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Server error: {str(e)}"
                })
                break
    
    except Exception as e:
        logger.error(f"WebSocket fatal error: {traceback.format_exc()}")
    finally:
        # Cleanup
        if session_id in sessions:
            del sessions[session_id]
        logger.info(f"WebSocket closed: {session_id}")
        try:
            await websocket.close()
        except:
            pass

@app.post("/chat")
async def chat_endpoint(message: str = Query(...), session_id: Optional[str] = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = SessionManager(session_id)
    
    session = sessions[session_id]
    
    response_chunks = []
    async for chunk in session.process_message(message):
        response_chunks.append(chunk)
    
    response = "".join(response_chunks)
    
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
                health_status[name] = {"status": "unhealthy", "error": str(e)}
    
    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status["redis"] = {"status": "healthy"}
        else:
            health_status["redis"] = {"status": "not configured"}
    except Exception as e:
        health_status["redis"] = {"status": "unhealthy", "error": str(e)}
    
    # Check vLLM
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{VLLM_URL}/health",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
            )
            health_status["vllm"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy"
            }
    except Exception as e:
        health_status["vllm"] = {"status": "unhealthy", "error": str(e)}
    
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
            "health": "/health",
            "models": {
                "info": "/api/models/info",
                "status": "/api/models/status",
                "config": "/api/models/config",
                "available": "/api/models/available"
            },
            "documents": {
                "upload": "/api/documents/upload",
                "search": "/api/documents/search",
                "list": "/api/documents/list",
                "stats": "/api/documents/stats"
            }
        }
    }

# ============= MODEL MANAGEMENT API =============

@app.get("/api/models/info")
async def get_vllm_model_info():
    """Get current model info from vLLM"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{VLLM_URL}/v1/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
            )
            response.raise_for_status()
            data = response.json()
            
            current_model = os.getenv('VLLM_MODEL', 'Unknown')
            data['current_config'] = {
                'model': current_model,
                'max_model_len': os.getenv('VLLM_MAX_MODEL_LEN', '4096'),
                'gpu_memory_utilization': os.getenv('VLLM_GPU_MEMORY', '0.9'),
                'quantization': os.getenv('VLLM_QUANTIZATION', 'awq')
            }
            
            return data
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to vLLM service")
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/status")
async def get_model_status():
    """Get current model status and system info"""
    try:
        vllm_healthy = False
        model_loaded = False
        current_model = os.getenv('VLLM_MODEL', 'Not configured')
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                health_response = await client.get(
                    f"{VLLM_URL}/health",
                    headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
                )
                vllm_healthy = health_response.status_code == 200
                
                if vllm_healthy:
                    models_response = await client.get(
                        f"{VLLM_URL}/v1/models",
                        headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
                    )
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        model_loaded = len(models_data.get("data", [])) > 0
                        if model_loaded and models_data["data"]:
                            current_model = models_data["data"][0].get("id", current_model)
        except Exception as e:
            logger.error(f"vLLM status check failed: {e}")
        
        gpu_enabled = os.path.exists('/dev/nvidia0') or 'CUDA_VISIBLE_DEVICES' in os.environ
        
        return {
            "vllm_healthy": vllm_healthy,
            "model_loaded": model_loaded,
            "current_model": current_model,
            "gpu_enabled": gpu_enabled,
            "vllm_url": VLLM_URL,
            "config": {
                "max_model_len": os.getenv('VLLM_MAX_MODEL_LEN', '4096'),
                "gpu_memory_utilization": float(os.getenv('VLLM_GPU_MEMORY', '0.9')),
                "quantization": os.getenv('VLLM_QUANTIZATION', 'awq')
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {"vllm_healthy": False, "error": str(e)}

@app.get("/api/models/available")
async def get_available_models():
    """Get list of recommended models for vLLM"""
    return {
        "recommended_12gb": [
            {
                "name": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                "size": "4.2GB",
                "description": "Excellent general-purpose model, fast inference",
                "quantization": "AWQ",
                "max_context": 32768
            },
            {
                "name": "TheBloke/Llama-2-7B-Chat-AWQ",
                "size": "3.9GB",
                "description": "Meta's conversational AI, good for chat",
                "quantization": "AWQ",
                "max_context": 4096
            },
            {
                "name": "TheBloke/neural-chat-7B-v3-3-AWQ",
                "size": "4.1GB",
                "description": "Intel's fine-tuned chat model",
                "quantization": "AWQ",
                "max_context": 8192
            }
        ],
        "small_models": [
            {
                "name": "microsoft/phi-2",
                "size": "2.7GB",
                "description": "Tiny but capable model",
                "quantization": "none",
                "max_context": 2048
            }
        ]
    }

@app.post("/api/models/config")
async def update_model_config(
    model: str = Query(..., description="Model name/path"),
    max_model_len: int = Query(4096, description="Maximum model length"),
    gpu_memory: float = Query(0.9, description="GPU memory utilization (0-1)")
):
    """Update vLLM configuration (requires container restart)"""
    try:
        os.environ['VLLM_MODEL'] = model
        os.environ['VLLM_MAX_MODEL_LEN'] = str(max_model_len)
        os.environ['VLLM_GPU_MEMORY'] = str(gpu_memory)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Configuration updated. Please restart vLLM container to apply changes.",
                "config": {
                    "model": model,
                    "max_model_len": max_model_len,
                    "gpu_memory": gpu_memory
                },
                "restart_command": "docker-compose restart vllm llm-service"
            }
        )
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= DOCUMENT/RAG MANAGEMENT API =============

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload document to RAG service"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(f"{RAG_SERVICE}/ingest", files=files)
            response.raise_for_status()
            
            result = response.json()
            
            if redis_client and result.get("status") == "success":
                doc_info = {
                    "id": str(uuid.uuid4()),
                    "name": file.filename,
                    "chunks": result.get("chunks_created", 0),
                    "uploaded": datetime.now().isoformat(),
                    "collection": result.get("collection", "documents")
                }
                
                redis_client.lpush("jarvis:documents", json.dumps(doc_info))
                redis_client.ltrim("jarvis:documents", 0, 99)
                result["document_id"] = doc_info["id"]
            
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

@app.get("/api/documents/list")
async def list_documents():
    """Get list of indexed documents"""
    try:
        if not redis_client:
            return {"documents": [], "message": "Redis not available"}
        
        doc_list = redis_client.lrange("jarvis:documents", 0, -1)
        documents = []
        
        for doc_json in doc_list:
            try:
                doc = json.loads(doc_json)
                documents.append(doc)
            except json.JSONDecodeError:
                continue
        
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"documents": [], "error": str(e)}

@app.get("/api/documents/stats")
async def get_document_stats():
    """Get RAG system statistics"""
    try:
        stats = {
            "rag_service": "unknown",
            "qdrant": "unknown",
            "total_documents": 0,
            "total_chunks": 0
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{RAG_SERVICE}/health")
                if response.status_code == 200:
                    stats["rag_service"] = "healthy"
                    health_data = response.json()
                    stats.update(health_data)
        except Exception:
            stats["rag_service"] = "unhealthy"
        
        if redis_client:
            try:
                doc_count = redis_client.llen("jarvis:documents")
                stats["total_documents"] = doc_count
            except Exception:
                pass
        
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}

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
