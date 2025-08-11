from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS LLM Service", version="1.0.0")

# Metrics
request_counter = Counter('llm_requests_total', 'Total LLM requests')
response_time = Histogram('llm_response_duration_seconds', 'LLM response time')

# Redis connection with error handling
redis_client = None
try:
    redis_host = os.getenv('REDIS_HOST', 'redis')
    redis_client = redis.Redis(
        host=redis_host,
        port=6379,
        decode_responses=True,
        socket_connect_timeout=5,
        retry_on_timeout=True,
        socket_keepalive=True,
        socket_keepalive_options={
            1: 1,  # TCP_KEEPIDLE
            2: 3,  # TCP_KEEPINTVAL  
            3: 5   # TCP_KEEPCNT
        }
    )
    redis_client.ping()
    logger.info(f"Connected to Redis at {redis_host}:6379")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# vLLM configuration
VLLM_HOST = os.getenv('VLLM_HOST', 'vllm')
VLLM_PORT = os.getenv('VLLM_PORT', '8000')
VLLM_API_KEY = os.getenv('VLLM_API_KEY', 'jarvis-key-123')
MODEL_NAME = os.getenv('MODEL_NAME', 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ')

class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True
    session_id: Optional[str] = None

class LLMEngine:
    def __init__(self):
        self.base_url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
        self.model = MODEL_NAME
        self.api_key = VLLM_API_KEY
        self.client = None
        self.vllm_available = False
        self.last_check_time = 0
        self.check_interval = 30  # Check vLLM availability every 30 seconds
    
    async def get_client(self):
        if not self.client:
            self.client = httpx.AsyncClient(
                timeout=120.0,
                headers={"Authorization": f"Bearer {self.api_key}"},
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self.client
    
    async def check_vllm_availability(self):
        """Check if vLLM is available with caching"""
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return self.vllm_available
            
        try:
            client = await self.get_client()
            response = await client.get(
                f"http://{VLLM_HOST}:{VLLM_PORT}/health",
                timeout=5.0
            )
            self.vllm_available = response.status_code == 200
            self.last_check_time = current_time
            if self.vllm_available:
                logger.info("vLLM service is available")
        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            self.vllm_available = False
            self.last_check_time = current_time
        
        return self.vllm_available
    
    async def generate(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        request_counter.inc()
        start_time = time.time()
        
        # Check vLLM availability first
        if not await self.check_vllm_availability():
            yield "Error: vLLM service is not available. Please ensure the vLLM container is running."
            return
        
        try:
            # Get context from Redis if session_id provided
            context = ""
            conversation_history = []
            
            if request.session_id and redis_client:
                try:
                    # Get conversation history
                    context_key = f"session:{request.session_id}:context"
                    context_data = redis_client.get(context_key)
                    if context_data:
                        context = json.loads(context_data)
                    
                    # Get last few messages for conversation history
                    history_key = f"session:{request.session_id}:history"
                    history_data = redis_client.lrange(history_key, -10, -1)
                    for msg in history_data:
                        try:
                            conversation_history.append(json.loads(msg))
                        except json.JSONDecodeError:
                            pass
                            
                except Exception as e:
                    logger.error(f"Redis error: {e}")
            
            # Build prompt for completions API (NOT chat completions)
            prompt = self._build_prompt(request, context, conversation_history)
            
            client = await self.get_client()
            
            # Store the full response for session history
            full_response = ""
            
            # Use vLLM's completions endpoint (NOT chat completions) - FIXED!
            try:
                if request.stream:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/completions",  # FIXED: Using /completions not /chat/completions
                        json={
                            "model": self.model,
                            "prompt": prompt,  # Single prompt string
                            "stream": True,
                            "temperature": request.temperature,
                            "max_tokens": request.max_tokens,
                            "top_p": 0.95,
                            "frequency_penalty": 0.0,
                            "presence_penalty": 0.0,
                        }
                    ) as response:
                        response.raise_for_status()
                        
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                line_data = line[6:]
                                if line_data == "[DONE]":
                                    break
                                try:
                                    data = json.loads(line_data)
                                    if "choices" in data and data["choices"]:
                                        text = data["choices"][0].get("text", "")
                                        if text:
                                            full_response += text
                                            yield text
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse: {line_data}, error: {e}")
                else:
                    # Non-streaming response
                    response = await client.post(
                        f"{self.base_url}/completions",  # FIXED: Using /completions
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "temperature": request.temperature,
                            "max_tokens": request.max_tokens,
                            "top_p": 0.95,
                        }
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    
                    if "choices" in response_data and response_data["choices"]:
                        full_response = response_data["choices"][0]["text"]
                        yield full_response
                        
            except httpx.ConnectError as e:
                logger.error(f"Connection error to vLLM: {e}")
                self.vllm_available = False
                yield f"Error: Cannot connect to vLLM service at {VLLM_HOST}:{VLLM_PORT}. Please check if the container is running."
                return
            
            # Store conversation in Redis for context
            if request.session_id and redis_client and full_response:
                try:
                    history_key = f"session:{request.session_id}:history"
                    conversation_turn = {
                        "timestamp": time.time(),
                        "user": request.prompt,
                        "assistant": full_response
                    }
                    redis_client.rpush(history_key, json.dumps(conversation_turn))
                    redis_client.ltrim(history_key, -20, -1)
                    redis_client.expire(history_key, 3600)
                    
                    context_key = f"session:{request.session_id}:context"
                    context_data = {
                        "last_interaction": time.time(),
                        "turns": redis_client.llen(history_key),
                        "last_topic": request.prompt[:100]
                    }
                    redis_client.setex(context_key, 3600, json.dumps(context_data))
                    
                except Exception as e:
                    logger.error(f"Failed to store conversation history: {e}")
                            
        except httpx.HTTPStatusError as e:
            # FIXED: Handle streaming response error properly
            error_text = "Unknown error"
            try:
                # For streaming responses, we need to read the content first
                error_text = await e.response.aread()
                error_text = error_text.decode('utf-8') if isinstance(error_text, bytes) else str(error_text)
            except:
                error_text = f"Status {e.response.status_code}"
            
            logger.error(f"vLLM API error: {error_text}")
            yield f"Error: vLLM service error - {e.response.status_code}"
            
        except httpx.TimeoutException as e:
            logger.error(f"vLLM timeout: {e}")
            yield "Error: Request to vLLM timed out. The model might be loading or overwhelmed."
            
        except Exception as e:
            logger.error(f"Generation error: {traceback.format_exc()}")
            yield f"Error: {str(e)}"
            
        finally:
            response_time.observe(time.time() - start_time)
    
    def _build_prompt(self, request: ChatRequest, context: Optional[str], history: list) -> str:
        """Build prompt string for completions API (not chat format)"""
        # System message
        system_content = request.system_prompt or """You are JARVIS, an advanced AI assistant inspired by Tony Stark's AI.
You are helpful, witty, and efficient. You provide concise but thorough responses.
You have a slightly sarcastic personality but remain professional and helpful.
Always be accurate and informative while maintaining an engaging conversational style."""
        
        # Build the prompt as a single string
        prompt_parts = []
        
        # Add system instruction
        prompt_parts.append(f"System: {system_content}\n")
        
        # Add context from RAG if available
        if context:
            prompt_parts.append(f"Context: {context}\n")
        
        # Add conversation history (last few turns for context)
        if history:
            prompt_parts.append("Previous conversation:\n")
            for turn in history[-3:]:  # Include last 3 turns
                if isinstance(turn, dict):
                    if "user" in turn:
                        prompt_parts.append(f"User: {turn['user']}\n")
                    if "assistant" in turn:
                        prompt_parts.append(f"Assistant: {turn['assistant']}\n")
        
        # Add current user message
        prompt_parts.append(f"User: {request.prompt}\n")
        prompt_parts.append("Assistant:")  # Prompt the model to respond
        
        return "".join(prompt_parts)
    
    async def cleanup(self):
        if self.client:
            await self.client.aclose()

engine = LLMEngine()

@app.get("/health")
async def health_check():
    try:
        # Check vLLM
        vllm_status = "unavailable"
        vllm_details = {"status": "checking"}
        model_info = {}
        
        if await engine.check_vllm_availability():
            try:
                client = await engine.get_client()
                
                # Get detailed vLLM status
                response = await client.get(
                    f"http://{VLLM_HOST}:{VLLM_PORT}/health",
                    timeout=5.0
                )
                if response.status_code == 200:
                    vllm_status = "healthy"
                    vllm_details = {"status": "healthy", "code": 200}
                    
                    # Try to get model info
                    try:
                        models_response = await client.get(
                            f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models",
                            timeout=5.0
                        )
                        if models_response.status_code == 200:
                            model_info = models_response.json()
                    except Exception as e:
                        logger.warning(f"Could not get model info: {e}")
                        
            except Exception as e:
                vllm_details = {"status": "error", "error": str(e)}
        else:
            vllm_details = {"status": "unavailable", "message": f"Cannot reach vLLM at {VLLM_HOST}:{VLLM_PORT}"}
        
        # Check Redis
        redis_status = "healthy"
        redis_details = {}
        if redis_client:
            try:
                redis_client.ping()
                redis_info = redis_client.info()
                redis_details = {
                    "status": "healthy",
                    "used_memory": redis_info.get("used_memory_human", "unknown"),
                    "connected_clients": redis_info.get("connected_clients", 0)
                }
            except Exception as e:
                redis_status = "unhealthy"
                redis_details = {"status": "unhealthy", "error": str(e)}
        else:
            redis_status = "not configured"
            redis_details = {"status": "not configured"}
        
        overall_status = "healthy" if vllm_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "model": MODEL_NAME,
            "services": {
                "vllm": vllm_details,
                "redis": redis_details
            },
            "model_info": model_info,
            "vllm_endpoint": f"http://{VLLM_HOST}:{VLLM_PORT}"
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
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            response = ""
            async for chunk in engine.generate(request):
                response += chunk
            return {
                "response": response,
                "model": MODEL_NAME,
                "session_id": request.session_id
            }
    except Exception as e:
        logger.error(f"Generate error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    vllm_status = "available" if await engine.check_vllm_availability() else "unavailable"
    return {
        "service": "JARVIS LLM Service",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "backend": "vLLM",
        "status": "online",
        "vllm_status": vllm_status,
        "vllm_endpoint": f"http://{VLLM_HOST}:{VLLM_PORT}"
    }

@app.post("/clear-session")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if not redis_client:
        return {"status": "redis not available"}
    
    try:
        keys_deleted = 0
        keys_deleted += redis_client.delete(f"session:{session_id}:history")
        keys_deleted += redis_client.delete(f"session:{session_id}:context")
        
        return {
            "status": "success",
            "session_id": session_id,
            "keys_deleted": keys_deleted
        }
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting LLM Service with vLLM model: {MODEL_NAME}")
    logger.info(f"vLLM endpoint: http://{VLLM_HOST}:{VLLM_PORT}")
    
    # Try to connect to vLLM with retries
    max_retries = 30
    for i in range(max_retries):
        try:
            if await engine.check_vllm_availability():
                logger.info("Connected to vLLM successfully")
                
                # Get model info
                try:
                    client = await engine.get_client()
                    models_response = await client.get(f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models")
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        logger.info(f"Available models: {models_data}")
                except Exception as e:
                    logger.warning(f"Could not get model list: {e}")
                break
        except Exception as e:
            logger.info(f"Waiting for vLLM... ({i+1}/{max_retries}): {e}")
            
        await asyncio.sleep(2)
    else:
        logger.warning("Could not connect to vLLM after maximum retries. Service will continue checking in background.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down LLM Service")
    await engine.cleanup()
