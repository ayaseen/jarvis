# services/orchestrator/app.py
from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
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
            "health": "/health",
            "models": {
                "status": "/api/models/status",
                "tags": "/api/models/tags",
                "pull": "/api/models/pull",
                "delete": "/api/models/delete",
                "set_active": "/api/models/set-active",
                "get_active": "/api/models/get-active"
            }
        }
    }

# ============= MODEL MANAGEMENT API =============

@app.get("/api/models/tags")
async def get_ollama_models():
    """Proxy to Ollama tags endpoint"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("http://ollama:11434/api/tags")
            response.raise_for_status()
            data = response.json()
            
            # Get the currently active model
            active_model = os.getenv('MODEL_NAME', 'mistral:latest')
            
            # Mark which model is active
            if data.get("models"):
                for model in data["models"]:
                    model["is_active"] = (model.get("name") == active_model)
            
            return data
    except httpx.HTTPError as e:
        logger.error(f"Failed to get models from Ollama: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/pull")
async def pull_ollama_model(model_name: str = Query(..., description="Model name to pull")):
    """Pull/download a model through Ollama"""
    logger.info(f"Pulling model: {model_name}")
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            # Send pull request to Ollama
            async with client.stream(
                "POST",
                "http://ollama:11434/api/pull",
                json={"name": model_name, "stream": True},
                timeout=None
            ) as response:
                response.raise_for_status()
                
                async def generate():
                    async for chunk in response.aiter_bytes():
                        yield chunk
                
                return StreamingResponse(
                    generate(),
                    media_type="application/x-ndjson",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no"
                    }
                )
    except httpx.HTTPError as e:
        logger.error(f"Failed to pull model {model_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to pull model: {str(e)}")
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/models/delete")
async def delete_ollama_model(model_name: str = Query(..., description="Model name to delete")):
    """Delete a model from Ollama"""
    logger.info(f"Deleting model: {model_name}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                "DELETE",
                "http://ollama:11434/api/delete",
                json={"name": model_name}
            )
            response.raise_for_status()
            return {"status": "success", "message": f"Model {model_name} deleted"}
    except httpx.HTTPError as e:
        logger.error(f"Failed to delete model {model_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to delete model: {str(e)}")
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/status")
async def get_model_status():
    """Get current model status"""
    try:
        # Check Ollama health
        ollama_healthy = False
        models_count = 0
        active_model = os.getenv('MODEL_NAME', 'mistral:latest')
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://ollama:11434/api/tags")
                if response.status_code == 200:
                    ollama_healthy = True
                    data = response.json()
                    models_count = len(data.get("models", []))
        except:
            pass
        
        # Check if GPU is actually being used
        gpu_enabled = os.path.exists('/dev/nvidia0') or 'CUDA_VISIBLE_DEVICES' in os.environ
        
        return {
            "ollama_healthy": ollama_healthy,
            "active_model": active_model,
            "models_count": models_count,
            "gpu_enabled": gpu_enabled
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "ollama_healthy": False,
            "error": str(e)
        }

@app.get("/api/models/get-active")
async def get_active_model():
    """Get the currently active model"""
    active_model = os.getenv('MODEL_NAME', 'mistral:latest')
    
    # Check if the active model is actually installed
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://ollama:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                installed_models = [m.get("name") for m in data.get("models", [])]
                
                if active_model in installed_models:
                    return {
                        "active_model": active_model,
                        "status": "ready",
                        "installed": True
                    }
                else:
                    return {
                        "active_model": active_model,
                        "status": "not_installed",
                        "installed": False,
                        "message": f"Model {active_model} is set as active but not installed"
                    }
    except Exception as e:
        logger.error(f"Error checking active model: {e}")
        return {
            "active_model": active_model,
            "status": "error",
            "error": str(e)
        }

@app.post("/api/models/set-active")
async def set_active_model(model_name: str = Query(..., description="Model name to set as active")):
    """Set the active model and restart LLM service"""
    logger.info(f"Setting active model to: {model_name}")
    
    try:
        # Check if model exists first
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://ollama:11434/api/tags")
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            
            if model_name not in models:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found. Please install it first.")
        
        # Update environment variable
        os.environ['MODEL_NAME'] = model_name
        
        # Also update the .env file if it exists
        env_file_path = '/app/.env'
        if os.path.exists(env_file_path):
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
        
        # Notify LLM service to reload (this would need to be implemented)
        # For now, return a message to restart manually
        return {
            "status": "success",
            "active_model": model_name,
            "message": f"Model set to {model_name}. Please run: docker-compose restart llm-service"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting active model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/available")
async def get_available_models():
    """Get list of models available to download"""
    return {
        "models": [
            {"name": "phi:latest", "size": "1.6GB", "description": "Microsoft's tiny but capable model"},
            {"name": "tinyllama:latest", "size": "638MB", "description": "Compact chat model"},
            {"name": "gemma:2b", "size": "1.4GB", "description": "Google's small model"},
            {"name": "mistral:latest", "size": "4.1GB", "description": "Excellent quality, fast"},
            {"name": "llama2:7b", "size": "3.8GB", "description": "Meta's general purpose"},
            {"name": "neural-chat:7b", "size": "4.1GB", "description": "Intel's conversational AI"},
            {"name": "llama2:13b", "size": "7.4GB", "description": "Larger Llama model"},
            {"name": "mixtral:8x7b-instruct-v0.1-q4_K_M", "size": "26GB", "description": "MoE architecture, very capable"},
            {"name": "solar:10.7b", "size": "6.1GB", "description": "Upstage's powerful model"},
        ]
    }

@app.post("/api/models/test")
async def test_model(model_name: str = Query(...), prompt: str = Query(default="Hello, how are you?")):
    """Test a specific model with a prompt"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
