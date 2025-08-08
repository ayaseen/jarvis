from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, ValidationError
import asyncio
import json
import os
from typing import Optional, AsyncGenerator
import httpx
import redis
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import logging
import traceback
# from langchain import LLMChain, PromptTemplate # This is not used, so it's good to remove it.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS LLM Service", version="1.0.0")

# Metrics
request_counter = Counter('llm_requests_total', 'Total LLM requests')
response_time = Histogram('llm_response_duration_seconds', 'LLM response time')

# Redis connection with error handling
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

# Ollama configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'ollama')
OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')
MODEL_NAME = os.getenv('MODEL_NAME', 'mixtral:8x7b-instruct-v0.1-q4_K_M')

class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True
    session_id: Optional[str] = None

class LLMEngine:
    def __init__(self):
        self.base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
        self.model = MODEL_NAME
        self.client = None
    
    async def get_client(self):
        if not self.client:
            self.client = httpx.AsyncClient(timeout=120.0)
        return self.client
    
    async def generate(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        request_counter.inc()
        start_time = time.time()
        
        try:
            context = ""
            if request.session_id and redis_client:
                try:
                    context_data = redis_client.get(f"session:{request.session_id}:context")
                    if context_data:
                        context = json.loads(context_data)
                except Exception as e:
                    logger.error(f"Redis error: {e}")
            
            full_prompt = self._build_prompt(request, context)
            
            client = await self.get_client()
            
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": request.stream,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                        "num_ctx": 4096,
                        "num_gpu": 50,
                    }
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse: {line}, error: {e}")
                            
        except Exception as e:
            logger.error(f"Generation error: {traceback.format_exc()}")
            yield f"Error: {str(e)}"
        finally:
            response_time.observe(time.time() - start_time)
    
    def _build_prompt(self, request: ChatRequest, context: Optional[str]) -> str:
        system = request.system_prompt or """You are JARVIS, an advanced AI assistant.
        You are helpful, witty, and efficient. Respond concisely but thoroughly."""
        
        prompt_parts = [system]
        if context:
            prompt_parts.append(f"Previous context:\n{context}")
        prompt_parts.append(f"User: {request.prompt}")
        prompt_parts.append("JARVIS:")
        
        return "\n\n".join(prompt_parts)
    
    async def cleanup(self):
        if self.client:
            await self.client.aclose()

engine = LLMEngine()

@app.get("/health")
async def health_check():
    try:
        client = await engine.get_client()
        response = await client.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags")
        response.raise_for_status()
        
        redis_status = "healthy"
        if redis_client:
            try:
                redis_client.ping()
            except:
                redis_status = "unhealthy"
        else:
            redis_status = "not configured"
        
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "redis": redis_status,
            "ollama": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/generate")
async def generate(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                engine.generate(request),
                media_type="text/event-stream"
            )
        else:
            response = ""
            async for chunk in engine.generate(request):
                response += chunk
            return {"response": response}
    except Exception as e:
        logger.error(f"Generate error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting LLM Service with model: {MODEL_NAME}")
    
    max_retries = 30
    for i in range(max_retries):
        try:
            client = await engine.get_client()
            response = await client.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags")
            if response.status_code == 200:
                logger.info("Connected to Ollama successfully")
                break
        except Exception as e:
            logger.info(f"Waiting for Ollama... ({i+1}/{max_retries}): {e}")
            await asyncio.sleep(2)

@app.on_event("shutdown")
async def shutdown_event():
    await engine.cleanup()
