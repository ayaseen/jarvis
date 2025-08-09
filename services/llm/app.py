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
    
    async def get_client(self):
        if not self.client:
            self.client = httpx.AsyncClient(
                timeout=120.0,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self.client
    
    async def generate(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        request_counter.inc()
        start_time = time.time()
        
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
                    history_data = redis_client.lrange(history_key, -10, -1)  # Last 10 messages
                    for msg in history_data:
                        try:
                            conversation_history.append(json.loads(msg))
                        except json.JSONDecodeError:
                            pass
                            
                except Exception as e:
                    logger.error(f"Redis error: {e}")
            
            # Build messages for vLLM chat completion
            messages = self._build_messages(request, context, conversation_history)
            
            client = await self.get_client()
            
            # Store the full response for session history
            full_response = ""
            
            # Use vLLM's OpenAI-compatible chat completions endpoint
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": request.stream,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "top_p": 0.95,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }
            ) as response:
                response.raise_for_status()
                
                if request.stream:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line_data = line[6:]  # Remove "data: " prefix
                            if line_data == "[DONE]":
                                break
                            try:
                                data = json.loads(line_data)
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_response += content
                                        yield content
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse: {line_data}, error: {e}")
                else:
                    # Non-streaming response
                    response_data = await response.json()
                    if "choices" in response_data and response_data["choices"]:
                        full_response = response_data["choices"][0]["message"]["content"]
                        yield full_response
            
            # Store conversation in Redis for context
            if request.session_id and redis_client and full_response:
                try:
                    # Store the conversation turn
                    history_key = f"session:{request.session_id}:history"
                    conversation_turn = {
                        "timestamp": time.time(),
                        "user": request.prompt,
                        "assistant": full_response
                    }
                    redis_client.rpush(history_key, json.dumps(conversation_turn))
                    
                    # Keep only last 20 turns
                    redis_client.ltrim(history_key, -20, -1)
                    
                    # Set expiry to 1 hour
                    redis_client.expire(history_key, 3600)
                    
                    # Update context summary (optional - for RAG integration)
                    context_key = f"session:{request.session_id}:context"
                    context_data = {
                        "last_interaction": time.time(),
                        "turns": redis_client.llen(history_key),
                        "last_topic": request.prompt[:100]  # Store topic hint
                    }
                    redis_client.setex(context_key, 3600, json.dumps(context_data))
                    
                except Exception as e:
                    logger.error(f"Failed to store conversation history: {e}")
                            
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM API error: {e.response.text}")
            yield f"Error: vLLM service error - {e.response.status_code}"
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to vLLM: {e}")
            yield "Error: Cannot connect to vLLM service. Please check if it's running."
        except Exception as e:
            logger.error(f"Generation error: {traceback.format_exc()}")
            yield f"Error: {str(e)}"
        finally:
            response_time.observe(time.time() - start_time)
    
    def _build_messages(self, request: ChatRequest, context: Optional[str], history: list) -> list:
        """Build messages array for OpenAI-compatible API with full context"""
        messages = []
        
        # System message
        system_content = request.system_prompt or """You are JARVIS, an advanced AI assistant inspired by Tony Stark's AI.
You are helpful, witty, and efficient. You provide concise but thorough responses.
You have a slightly sarcastic personality but remain professional and helpful.
Always be accurate and informative while maintaining an engaging conversational style."""
        
        messages.append({"role": "system", "content": system_content})
        
        # Add context from RAG if available
        if context:
            messages.append({
                "role": "system", 
                "content": f"Relevant context from knowledge base:\n{context}"
            })
        
        # Add conversation history (last few turns for context)
        for turn in history[-5:]:  # Include last 5 turns
            if isinstance(turn, dict):
                if "user" in turn:
                    messages.append({"role": "user", "content": turn["user"]})
                if "assistant" in turn:
                    messages.append({"role": "assistant", "content": turn["assistant"]})
        
        # Add current user message
        messages.append({"role": "user", "content": request.prompt})
        
        return messages
    
    async def cleanup(self):
        if self.client:
            await self.client.aclose()

engine = LLMEngine()

@app.get("/health")
async def health_check():
    try:
        client = await engine.get_client()
        
        # Check vLLM health endpoint
        try:
            response = await client.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health")
            vllm_status = "healthy" if response.status_code == 200 else "unhealthy"
            vllm_details = {"status": vllm_status, "code": response.status_code}
        except httpx.ConnectError:
            vllm_status = "unreachable"
            vllm_details = {"status": "unreachable", "error": "Cannot connect to vLLM"}
        except Exception as e:
            vllm_status = "error"
            vllm_details = {"status": "error", "error": str(e)}
        
        # Check Redis
        redis_status = "healthy"
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
        
        # Get model info if vLLM is healthy
        model_info = {}
        if vllm_status == "healthy":
            try:
                models_response = await client.get(f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models")
                if models_response.status_code == 200:
                    model_info = models_response.json()
            except Exception as e:
                logger.warning(f"Could not get model info: {e}")
        
        overall_status = "healthy" if vllm_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "model": MODEL_NAME,
            "services": {
                "vllm": vllm_details,
                "redis": redis_details
            },
            "model_info": model_info,
            "device": engine.device if hasattr(engine, 'device') else "unknown"
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
    return {
        "service": "JARVIS LLM Service",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "backend": "vLLM",
        "status": "online"
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
    
    # Wait for vLLM to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            client = await engine.get_client()
            response = await client.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health")
            if response.status_code == 200:
                logger.info("Connected to vLLM successfully")
                
                # Get model info
                try:
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
        logger.warning("Could not connect to vLLM after maximum retries. Service will start but may not function properly.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down LLM Service")
    await engine.cleanup()
