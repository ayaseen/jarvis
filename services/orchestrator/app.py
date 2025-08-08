# services/orchestrator/app.py
from fastapi import FastAPI, WebSocket, HTTPException, Query
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
OLLAMA_URL = 'http://ollama:11434'

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
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                health_status["ollama"] = {"status": "healthy"}
            else:
                health_status["ollama"] = {"status": "unhealthy"}
    except Exception as e:
        health_status["ollama"] = {"status": "unhealthy", "error": str(e)}
    
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
                "status": "/api/models/status",
                "tags": "/api/models/tags",
                "pull": "/api/models/pull",
                "delete": "/api/models/delete",
                "set_active": "/api/models/set-active"
            }
        }
    }

# ============= MODEL MANAGEMENT API =============

@app.get("/api/models/tags")
async def get_ollama_models():
    """Get list of installed models from Ollama"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            # Get the currently active model
            active_model = os.getenv('MODEL_NAME', 'mixtral:8x7b-instruct-v0.1-q4_K_M')
            
            # Mark which model is active
            if data.get("models"):
                for model in data["models"]:
                    model["is_active"] = (model.get("name") == active_model)
            
            return data
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama service")
    except httpx.HTTPError as e:
        logger.error(f"Failed to get models from Ollama: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama service error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/pull")
async def pull_ollama_model(model_name: str = Query(..., description="Model name to pull")):
    """Pull/download a model through Ollama with proper streaming"""
    logger.info(f"Pulling model: {model_name}")
    
    try:
        # Create a very long timeout for large model downloads
        timeout = httpx.Timeout(7200.0, connect=60.0)  # 2 hours total, 1 minute connect
        
        async def stream_generator():
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    # Make the request to Ollama
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_URL}/api/pull",
                        json={"name": model_name, "stream": True}
                    ) as response:
                        response.raise_for_status()
                        
                        # Stream the response chunks
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
                                
                except httpx.ConnectError:
                    error_data = json.dumps({
                        "error": "Cannot connect to Ollama service",
                        "status": "Connection failed - is Ollama running?"
                    })
                    yield error_data.encode() + b'\n'
                except httpx.TimeoutException:
                    error_data = json.dumps({
                        "error": "Download timeout",
                        "status": "Timeout - model too large, use manual pull"
                    })
                    yield error_data.encode() + b'\n'
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    error_data = json.dumps({
                        "error": str(e),
                        "status": f"Error: {str(e)}"
                    })
                    yield error_data.encode() + b'\n'
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error initiating pull: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/models/delete")
async def delete_ollama_model(model_name: str = Query(..., description="Model name to delete")):
    """Delete a model from Ollama"""
    logger.info(f"Deleting model: {model_name}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                "DELETE",
                f"{OLLAMA_URL}/api/delete",
                json={"name": model_name}
            )
            response.raise_for_status()
            return {"status": "success", "message": f"Model {model_name} deleted"}
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama for delete: {e}")
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama service")
    except httpx.HTTPError as e:
        logger.error(f"Failed to delete model {model_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to delete model: {str(e)}")
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/status")
async def get_model_status():
    """Get current model status and system info"""
    try:
        ollama_healthy = False
        models_count = 0
        active_model = os.getenv('MODEL_NAME', 'mixtral:8x7b-instruct-v0.1-q4_K_M')
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{OLLAMA_URL}/api/tags")
                if response.status_code == 200:
                    ollama_healthy = True
                    data = response.json()
                    models_count = len(data.get("models", []))
        except Exception as e:
            logger.error(f"Ollama status check failed: {e}")
        
        # Check if GPU is available
        gpu_enabled = os.path.exists('/dev/nvidia0') or 'CUDA_VISIBLE_DEVICES' in os.environ
        
        return {
            "ollama_healthy": ollama_healthy,
            "active_model": active_model,
            "models_count": models_count,
            "gpu_enabled": gpu_enabled,
            "ollama_url": OLLAMA_URL
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "ollama_healthy": False,
            "error": str(e)
        }

@app.post("/api/models/set-active")
async def set_active_model(model_name: str = Query(..., description="Model name to set as active")):
    """Set the active model for the LLM service"""
    logger.info(f"Setting active model to: {model_name}")
    
    try:
        # Check if model exists first
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            
            if model_name not in models:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model {model_name} not found. Please install it first."
                )
        
        # Update environment variable
        os.environ['MODEL_NAME'] = model_name
        
        # Try to update .env file if it exists
        env_file_path = '/app/.env'
        if os.path.exists(env_file_path):
            try:
                with open(env_file_path, 'r') as f:
                    lines = f.readlines()
                
                with open(env_file_path, 'w') as f:
                    model_found = False
                    for line in lines:
                        if line.startswith('MODEL_NAME='):
                            f.write(f'MODEL_NAME={model_name}\n')
                            model_found = True
                        else:
                            f.write(line)
                    
                    if not model_found:
                        f.write(f'MODEL_NAME={model_name}\n')
            except Exception as e:
                logger.warning(f"Could not update .env file: {e}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "active_model": model_name,
                "message": f"Model set to {model_name}. Please run: docker-compose restart llm-service"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting active model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/api/models/{path:path}")
async def options_handler():
    """Handle OPTIONS requests for CORS"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/api/models/test-pull")
async def test_pull_connectivity():
    """Test if we can connect to Ollama for pulling models"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Just test if Ollama is reachable
            response = await client.get(f"{OLLAMA_URL}/api/version")
            if response.status_code == 200:
                return {"status": "ready", "ollama": "connected", "can_pull": True}
            else:
                return {"status": "error", "ollama": "unreachable", "can_pull": False}
    except Exception as e:
        return {"status": "error", "error": str(e), "can_pull": False}

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
