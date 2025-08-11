# services/orchestrator/app.py - COMPLETE PRODUCTION VERSION
from fastapi import FastAPI, WebSocket, HTTPException, Query, File, UploadFile, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
import httpx
import json
import asyncio
from typing import Dict, Optional, List, AsyncGenerator, Any
import redis
import uuid
from datetime import datetime
import logging
import os
import traceback
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS Orchestrator", version="3.0.0")

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

# Document storage path
DOCUMENTS_PATH = '/mnt/appsdata/jarvis/documents'
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

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

# Service health cache
health_cache = {
    "last_check": 0,
    "cache_duration": 10,  # seconds
    "data": None
}

async def check_service_health(service_url: str, service_name: str, timeout: float = 5.0) -> tuple:
    """Check health of a service with proper error handling"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{service_url}/health")
            if response.status_code == 200:
                data = response.json()
                # Extract additional info based on service
                if service_name == "rag" and "milvus_status" in data:
                    return True, data.get("milvus_status") == "connected"
                return True, None
            return False, None
    except Exception as e:
        logger.error(f"{service_name} health check failed: {e}")
        return False, None

@app.get("/")
async def root():
    return {
        "message": "JARVIS Orchestrator Service",
        "version": "3.0.0",
        "status": "operational",
        "endpoints": {
            "websocket": "/ws",
            "chat": "/api/chat",
            "stream": "/api/chat/stream",
            "models": "/api/models",
            "health": "/health",
            "api_health": "/api/health",
            "documents": "/api/documents",
            "collections": "/api/collections"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with caching"""
    current_time = time.time()
    
    # Use cached data if available and fresh
    if health_cache["data"] and (current_time - health_cache["last_check"]) < health_cache["cache_duration"]:
        return health_cache["data"]
    
    services_status = {}
    
    # Check all services in parallel
    results = await asyncio.gather(
        check_service_health(LLM_SERVICE, "llm"),
        check_service_health(RAG_SERVICE, "rag"),
        check_service_health(VOICE_SERVICE, "voice"),
        check_service_health(VLLM_URL, "vllm"),
        return_exceptions=True
    )
    
    # Process results
    services_status["llm"] = results[0][0] if not isinstance(results[0], Exception) else False
    
    # RAG and Milvus status
    if not isinstance(results[1], Exception):
        services_status["rag"] = results[1][0]
        services_status["milvus"] = results[1][1] if results[1][1] is not None else False
    else:
        services_status["rag"] = False
        services_status["milvus"] = False
    
    services_status["voice"] = results[2][0] if not isinstance(results[2], Exception) else False
    services_status["vllm"] = results[3][0] if not isinstance(results[3], Exception) else False
    
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
    
    response_data = {
        "status": "healthy" if all_healthy else "degraded",
        "services": services_status,
        "timestamp": datetime.utcnow().isoformat(),
        "active_connections": len(active_connections)
    }
    
    # Cache the response
    health_cache["data"] = response_data
    health_cache["last_check"] = current_time
    
    return response_data

# CRITICAL: Add /api/health endpoint that document.html calls
@app.get("/api/health")
async def api_health_check():
    """API health check endpoint - used by document.html"""
    return await health_check()

@app.get("/api/models")
async def list_models():
    """List available models from vLLM"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{VLLM_URL}/v1/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
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

@app.get("/api/models/status")
async def model_status():
    """Get current model status"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{VLLM_URL}/v1/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    return {
                        "model_loaded": True,
                        "current_model": models[0]["id"]
                    }
    except Exception as e:
        logger.error(f"Model status check failed: {e}")
    
    return {"model_loaded": False, "current_model": None}

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Query(default="documents")
):
    """Upload document for RAG indexing with comprehensive error handling"""
    logger.info(f"Received upload request for file: {file.filename}, collection: {collection}")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Save file to documents directory
        file_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
        safe_filename = f"{file_hash}_{file.filename}"
        file_path = os.path.join(DOCUMENTS_PATH, safe_filename)
        
        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Save file locally
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved file: {safe_filename} to {file_path} ({len(file_content)} bytes)")
        
        # Send to RAG service for indexing
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Reset file for reading
            await file.seek(0)
            
            files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
            
            # Send to RAG service with collection parameter
            logger.info(f"Sending to RAG service: {RAG_SERVICE}/ingest?collection={collection}")
            
            response = await client.post(
                f"{RAG_SERVICE}/ingest",
                files=files,
                params={"collection": collection}
            )
            
            logger.info(f"RAG service response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"RAG indexing successful: {result}")
            
            # Store document metadata in database
            try:
                conn = get_db_connection()
                if conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO documents (filename, content, metadata, created_at)
                            VALUES (%s, %s, %s, NOW())
                            ON CONFLICT DO NOTHING
                        """, (
                            file.filename,
                            file_content.decode('utf-8', errors='ignore')[:1000],  # Store first 1000 chars
                            json.dumps({
                                "size": len(file_content),
                                "chunks": result.get("chunks_created", 0),
                                "collection": collection,
                                "path": file_path,
                                "hash": file_hash,
                                "original_filename": file.filename
                            })
                        ))
                        conn.commit()
                        logger.info(f"Stored document metadata in database")
                    conn.close()
            except Exception as e:
                logger.error(f"Failed to store document metadata: {e}")
                # Don't fail the request if DB storage fails
            
            return {
                "status": "success",
                "filename": file.filename,
                "chunks_created": result.get("chunks_created", 0),
                "collection": collection,
                "document_id": result.get("document_id"),
                "file_hash": result.get("file_hash", file_hash),
                "path": file_path,
                "text_length": result.get("text_length", 0),
                "sample_text": result.get("sample_text", "")
            }
        else:
            # Get error details
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", str(error_data))
            except:
                error_detail = response.text or f"HTTP {response.status_code}"
            
            logger.error(f"RAG indexing failed: {error_detail}")
            
            # Clean up file if indexing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            
            raise HTTPException(
                status_code=500,
                detail=f"Document indexing failed: {error_detail}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}\n{traceback.format_exc()}")
        
        # Clean up file on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents(limit: int = Query(default=50)):
    """List indexed documents with complete information"""
    documents = []
    total_chunks = 0
    
    try:
        # Get collection stats from RAG service first
        collection_stats = {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                collections_response = await client.get(f"{RAG_SERVICE}/collections")
                if collections_response.status_code == 200:
                    collections_data = collections_response.json()
                    for detail in collections_data.get("details", []):
                        collection_stats[detail.get("name")] = detail.get("num_entities", 0)
                    total_chunks = sum(collection_stats.values())
                    logger.info(f"Collection stats: {collection_stats}")
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
        
        # Get documents from database
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT filename, metadata, created_at 
                    FROM documents 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
                db_docs = cursor.fetchall()
                
                for doc in db_docs:
                    metadata = doc["metadata"] or {}
                    documents.append({
                        "filename": doc["filename"],
                        "metadata": metadata,
                        "created_at": doc["created_at"].isoformat() if doc["created_at"] else None
                    })
            conn.close()
        
        # Also list files from documents directory
        if os.path.exists(DOCUMENTS_PATH):
            existing_filenames = {d["filename"] for d in documents}
            for filename in os.listdir(DOCUMENTS_PATH):
                file_path = os.path.join(DOCUMENTS_PATH, filename)
                if os.path.isfile(file_path):
                    # Extract original filename from safe filename
                    original_name = filename
                    if "_" in filename and len(filename) > 9:
                        # Remove hash prefix
                        original_name = filename[9:] if filename[8] == "_" else filename
                    
                    # Check if already in database results
                    if original_name not in existing_filenames:
                        stat = os.stat(file_path)
                        documents.append({
                            "filename": original_name,
                            "metadata": {
                                "size": stat.st_size,
                                "collection": "documents",
                                "chunks": 0  # Unknown for files not in DB
                            },
                            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
                        })
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}\n{traceback.format_exc()}")
    
    return {
        "documents": documents[:limit], 
        "total": len(documents),
        "total_chunks": total_chunks,
        "collections": collection_stats if 'collection_stats' in locals() else {}
    }

@app.get("/api/collections")
async def list_collections():
    """Proxy to RAG service collections endpoint with error handling"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RAG_SERVICE}/collections")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Collections data: {data}")
                return data
    except Exception as e:
        logger.error(f"Failed to get collections: {e}")
    
    return {"collections": [], "details": [], "total": 0, "error": "Could not fetch collections"}

@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection via RAG service"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.delete(f"{RAG_SERVICE}/collections/{collection_name}/clear")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test/rag")
async def test_rag(query: str = Query(...)):
    """Test RAG search functionality with detailed logging"""
    logger.info(f"Testing RAG with query: {query}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_SERVICE}/search",
                json={
                    "query": query,
                    "collection": "documents",
                    "top_k": 5,
                    "threshold": 0.3
                }
            )
            
            logger.info(f"RAG test response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"RAG test results: {data.get('count')} results found")
                return data
    except Exception as e:
        logger.error(f"RAG test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"results": [], "count": 0, "error": "RAG service unavailable"}

@app.post("/api/chat")
async def chat_endpoint(request: dict):
    """Synchronous chat endpoint with RAG integration"""
    try:
        message = request.get("message", "")
        use_rag = request.get("use_rag", True)
        session_id = request.get("session_id", str(uuid.uuid4()))
        collection = request.get("collection", "documents")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        logger.info(f"Chat request: message='{message[:50]}...', use_rag={use_rag}, collection={collection}")
        
        # Get context from session
        context = session_contexts.get(session_id, [])
        
        # RAG search if enabled
        context_docs = []
        context_used = False
        context_count = 0
        
        if use_rag:
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    logger.info(f"Searching RAG for: {message}")
                    response = await client.post(
                        f"{RAG_SERVICE}/search",
                        json={
                            "query": message,
                            "collection": collection,
                            "top_k": 3,
                            "threshold": 0.3  # Lower threshold for better recall
                        }
                    )
                    if response.status_code == 200:
                        result = response.json()
                        context_docs = result.get("results", [])
                        context_count = len(context_docs)
                        context_used = context_count > 0
                        logger.info(f"RAG found {context_count} documents")
                        if context_docs:
                            logger.info(f"Top result: {context_docs[0].get('text', '')[:100]}...")
                except Exception as e:
                    logger.error(f"RAG search failed: {e}")
        
        # Prepare prompt with context
        prompt = message
        if context_docs:
            context_text = "\n\n".join([
                f"[Document {i+1}]: {doc.get('text', '')}"
                for i, doc in enumerate(context_docs)
            ])
            prompt = f"Use the following context to answer the question:\n\n{context_text}\n\nQuestion: {message}\n\nAnswer:"
        
        # Generate response using LLM service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{LLM_SERVICE}/generate",
                json={
                    "prompt": prompt,
                    "session_id": session_id,
                    "stream": False,
                    "temperature": 0.7,
                    "max_tokens": 512
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                
                # Update session context
                context.append({"user": message, "assistant": answer})
                session_contexts[session_id] = context[-10:]  # Keep last 10 exchanges
                
                return {
                    "response": answer,
                    "session_id": session_id,
                    "context_used": context_used,
                    "context_count": context_count
                }
            else:
                error_msg = f"LLM service returned status {response.status_code}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail="LLM generation failed")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket
    logger.info(f"WebSocket connection established: {session_id}")
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "session_id": session_id,
            "message": "Connected to JARVIS. How can I assist you?"
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            # Handle ping/pong for keepalive
            if data == '{"type":"ping"}':
                await websocket.send_json({"type": "pong"})
                continue
            
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            use_rag = message_data.get("use_rag", True)
            
            if not user_message:
                continue
            
            logger.info(f"WebSocket message: {user_message[:50]}...")
            
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
                                    "threshold": 0.3
                                }
                            )
                            if response.status_code == 200:
                                context_docs = response.json().get("results", [])
                                logger.info(f"WebSocket RAG found {len(context_docs)} docs")
                        except Exception as e:
                            logger.error(f"WebSocket RAG search failed: {e}")
                
                # Prepare prompt
                prompt = user_message
                if context_docs:
                    context_text = "\n\n".join([doc.get("text", "") for doc in context_docs])
                    prompt = f"Context:\n{context_text}\n\nQuestion: {user_message}\n\nAnswer:"
                
                # Stream response from LLM
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream(
                        "POST",
                        f"{LLM_SERVICE}/generate",
                        json={
                            "prompt": prompt,
                            "session_id": session_id,
                            "stream": True,
                            "temperature": 0.7,
                            "max_tokens": 512
                        }
                    ) as response:
                        full_response = ""
                        async for chunk in response.aiter_text():
                            if chunk:
                                full_response += chunk
                                # Send chunk to client
                                await websocket.send_json({
                                    "type": "chunk",
                                    "content": chunk
                                })
                        
                        # Send completion message
                        await websocket.send_json({
                            "type": "complete",
                            "message": full_response,
                            "context_used": len(context_docs) > 0
                        })
                        
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

@app.post("/api/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text using voice service"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"audio": (file.filename, await file.read(), file.content_type)}
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

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    logger.info("=" * 50)
    logger.info("JARVIS Orchestrator Starting")
    logger.info(f"LLM Service: {LLM_SERVICE}")
    logger.info(f"RAG Service: {RAG_SERVICE}")
    logger.info(f"Voice Service: {VOICE_SERVICE}")
    logger.info(f"vLLM: {VLLM_URL}")
    logger.info(f"Documents Path: {DOCUMENTS_PATH}")
    logger.info("=" * 50)
    
    # Test connections
    health = await health_check()
    logger.info(f"Initial health check: {health}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("JARVIS Orchestrator shutting down")
    # Close all websocket connections
    for session_id, ws in list(active_connections.items()):
        try:
            await ws.close()
        except:
            pass
    active_connections.clear()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)
