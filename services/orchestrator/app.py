# services/orchestrator/app.py
from fastapi import FastAPI, WebSocket, HTTPException, Query, File, UploadFile
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
    
    async def process_message(self, message: str, use_rag: bool = True) -> AsyncGenerator[str, None]:
        context_docs = []
        if use_rag:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
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
        
        # Store context
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

# Session storage
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
    
    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status["redis"] = {"status": "healthy"}
        else:
            health_status["redis"] = {"status": "unhealthy", "message": "Redis client is not configured"}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["redis"] = {"status": "unhealthy"}
    
    # Check vLLM
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{VLLM_URL}/health",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
            )
            if response.status_code == 200:
                health_status["vllm"] = {"status": "healthy"}
            else:
                health_status["vllm"] = {"status": "unhealthy"}
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
            }
        }
    }

# ============= MODEL MANAGEMENT API FOR vLLM =============

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
            
            # Add additional info
            current_model = os.getenv('VLLM_MODEL', 'Unknown')
            data['current_config'] = {
                'model': current_model,
                'max_model_len': os.getenv('VLLM_MAX_MODEL_LEN', '4096'),
                'gpu_memory_utilization': os.getenv('VLLM_GPU_MEMORY', '0.9'),
                'quantization': os.getenv('VLLM_QUANTIZATION', 'awq')
            }
            
            return data
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to vLLM: {e}")
        raise HTTPException(status_code=503, detail="Cannot connect to vLLM service")
    except httpx.HTTPError as e:
        logger.error(f"Failed to get model info from vLLM: {e}")
        raise HTTPException(status_code=503, detail=f"vLLM service error: {str(e)}")
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
                # Check health
                health_response = await client.get(
                    f"{VLLM_URL}/health",
                    headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
                )
                vllm_healthy = health_response.status_code == 200
                
                # Check loaded model
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
        
        # Check if GPU is available
        gpu_enabled = os.path.exists('/dev/nvidia0') or 'CUDA_VISIBLE_DEVICES' in os.environ
        
        return {
            "vllm_healthy": vllm_healthy,
            "model_loaded": model_loaded,
            "current_model": current_model,
            "gpu_enabled": gpu_enabled,
            "vllm_url": VLLM_URL,
            "config": {
                "max_model_len": os.getenv('VLLM_MAX_MODEL_LEN', '4096'),
                "gpu_memory_utilization": os.getenv('VLLM_GPU_MEMORY', '0.9'),
                "quantization": os.getenv('VLLM_QUANTIZATION', 'awq')
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "vllm_healthy": False,
            "error": str(e)
        }

@app.get("/api/models/available")
async def get_available_models():
    """Get list of recommended models for vLLM with 12GB VRAM"""
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
            },
            {
                "name": "TheBloke/zephyr-7B-beta-AWQ",
                "size": "4.2GB",
                "description": "HuggingFace's aligned chat model",
                "quantization": "AWQ",
                "max_context": 32768
            },
            {
                "name": "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ",
                "size": "4.2GB",
                "description": "High-quality instruction-following model",
                "quantization": "AWQ",
                "max_context": 32768
            },
            {
                "name": "TheBloke/Starling-LM-7B-alpha-AWQ",
                "size": "4.2GB",
                "description": "Berkeley's RLHF model, very capable",
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
            },
            {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "size": "1.1GB",
                "description": "Very small, fast model",
                "quantization": "none",
                "max_context": 2048
            }
        ],
        "notes": {
            "awq_models": "AWQ quantization provides best balance of quality and speed for RTX 4070 Ti",
            "memory_usage": "With 12GB VRAM, 7B AWQ models leave room for context and batch processing",
            "performance": "Expect 50-100+ tokens/sec with AWQ models on RTX 4070 Ti"
        }
    }

@app.post("/api/models/config")
async def update_model_config(
    model: str = Query(..., description="Model name/path"),
    max_model_len: int = Query(4096, description="Maximum model length"),
    gpu_memory: float = Query(0.9, description="GPU memory utilization (0-1)")
):
    """Update vLLM configuration (requires container restart)"""
    try:
        # Update environment variables
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

@app.options("/api/models/{path:path}")
async def options_handler():
    """Handle OPTIONS requests for CORS"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Additional error handling
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

# ============= DOCUMENT/RAG MANAGEMENT API =============

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Proxy document upload to RAG service"""
    try:
        # Forward the file to RAG service
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(
                f"{RAG_SERVICE}/ingest",
                files=files
            )
            response.raise_for_status()

            result = response.json()

            # Store document info in Redis if available
            if redis_client and result.get("status") == "success":
                doc_info = {
                    "id": str(uuid.uuid4()),
                    "name": file.filename,
                    "chunks": result.get("chunks_created", 0),
                    "uploaded": datetime.now().isoformat(),
                    "collection": result.get("collection", "documents")
                }

                # Store in Redis list
                redis_client.lpush("jarvis:documents", json.dumps(doc_info))
                redis_client.ltrim("jarvis:documents", 0, 99)  # Keep last 100 docs

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

    except httpx.HTTPStatusError as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/list")
async def list_documents():
    """Get list of indexed documents from Redis"""
    try:
        if not redis_client:
            return {"documents": [], "message": "Redis not available"}

        # Get documents from Redis
        doc_list = redis_client.lrange("jarvis:documents", 0, -1)
        documents = []

        for doc_json in doc_list:
            try:
                doc = json.loads(doc_json)
                documents.append(doc)
            except json.JSONDecodeError:
                continue

        # Get stats from Qdrant
        stats = {}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://qdrant:6333/collections")
                if response.status_code == 200:
                    collections_data = response.json()
                    stats["collections"] = len(collections_data.get("collections", []))

                    # Get document collection info
                    coll_response = await client.get(f"http://qdrant:6333/collections/documents")
                    if coll_response.status_code == 200:
                        coll_data = coll_response.json()
                        stats["total_vectors"] = coll_data.get("result", {}).get("vectors_count", 0)
        except Exception as e:
            logger.warning(f"Could not get Qdrant stats: {e}")

        return {
            "documents": documents,
            "total": len(documents),
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"documents": [], "error": str(e)}

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Remove a document from the index"""
    try:
        # Remove from Redis list
        if redis_client:
            doc_list = redis_client.lrange("jarvis:documents", 0, -1)
            for doc_json in doc_list:
                doc = json.loads(doc_json)
                if doc.get("id") == document_id:
                    redis_client.lrem("jarvis:documents", 1, doc_json)
                    break

        # Note: Actual vector removal from Qdrant would require storing point IDs
        # This is a simplified version that just removes from the document list

        return {"status": "success", "message": f"Document {document_id} removed from list"}

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/stats")
async def get_document_stats():
    """Get RAG system statistics"""
    try:
        stats = {
            "rag_service": "unknown",
            "qdrant": "unknown",
            "total_documents": 0,
            "total_chunks": 0,
            "collections": 0
        }

        # Check RAG service
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{RAG_SERVICE}/health")
                if response.status_code == 200:
                    stats["rag_service"] = "healthy"
                    health_data = response.json()
                    stats.update(health_data)
        except Exception:
            stats["rag_service"] = "unhealthy"

        # Check Qdrant
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://qdrant:6333/collections")
                if response.status_code == 200:
                    stats["qdrant"] = "healthy"
                    collections = response.json().get("collections", [])
                    stats["collections"] = len(collections)

                    # Get documents collection stats
                    for collection in collections:
                        if collection["name"] == "documents":
                            stats["total_chunks"] = collection.get("vectors_count", 0)
        except Exception:
            stats["qdrant"] = "unhealthy"

        # Get document count from Redis
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
