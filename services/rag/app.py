from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import PyPDF2
import docx
from pathlib import Path
import logging
import torch
import uuid
import traceback
import asyncio
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS RAG Service", version="1.0.0")

# Service configuration with retry logic
QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')
TRANSFORMERS_CACHE = os.getenv('TRANSFORMERS_CACHE', '/models/transformers')

class RAGRequest(BaseModel):
    query: str
    collection: str = "documents"
    top_k: int = 5
    threshold: float = 0.7

class RAGEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Set cache directory for models
        os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
        os.environ['HF_HOME'] = TRANSFORMERS_CACHE
        
        try:
            self.embedder = SentenceTransformer(
                EMBEDDING_MODEL, 
                device=self.device,
                cache_folder=TRANSFORMERS_CACHE
            )
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        self.qdrant = None
        self.connection_retries = 0
        self.max_retries = 30

    async def connect_to_qdrant(self):
        """Connect to Qdrant with retry logic"""
        while self.connection_retries < self.max_retries:
            try:
                self.qdrant = QdrantClient(
                    host=QDRANT_HOST, 
                    port=QDRANT_PORT, 
                    timeout=30,
                    grpc_port=None,  # Disable gRPC
                    prefer_grpc=False  # Use HTTP
                )
                # Test connection
                self.qdrant.get_collections()
                logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
                self._init_collections()
                return
            except Exception as e:
                self.connection_retries += 1
                logger.warning(f"Qdrant connection attempt {self.connection_retries}/{self.max_retries} failed: {e}")
                if self.connection_retries >= self.max_retries:
                    logger.error("Max retries reached. Qdrant connection failed.")
                    self.qdrant = None
                    return
                await asyncio.sleep(2)
    
    def _init_collections(self):
        """Initialize collections with error handling"""
        if not self.qdrant:
            logger.warning("Qdrant client not initialized")
            return
            
        collections = {
            "documents": 768,
            "conversations": 768,
            "knowledge": 768
        }
        
        for name, dim in collections.items():
            try:
                # Check if collection exists
                existing_collections = self.qdrant.get_collections()
                exists = any(col.name == name for col in existing_collections.collections)
                
                if not exists:
                    self.qdrant.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                    )
                    logger.info(f"Created collection: {name}")
                else:
                    logger.info(f"Collection already exists: {name}")
            except Exception as e:
                logger.warning(f"Collection {name} initialization error: {e}")
    
    async def ingest_file(self, file: UploadFile) -> Dict[str, Any]:
        if not self.qdrant:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
            
        temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        
        try:
            # Save file
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Extract text
            text = self._extract_text(temp_path, file.filename)
            
            if not text or len(text.strip()) < 10:
                raise ValueError("No text content extracted from file")
            
            # Chunk text
            chunks = self._chunk_text(text)
            
            if not chunks:
                raise ValueError("No chunks created from text")
            
            # Create embeddings and store
            points = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.embedder.encode(chunk).tolist()
                    
                    # Generate unique ID
                    chunk_id = str(uuid.uuid4())
                    
                    points.append(PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source": file.filename,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    ))
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
            
            if points:
                self.qdrant.upsert(
                    collection_name="documents",
                    points=points
                )
            
            return {
                "status": "success",
                "filename": file.filename,
                "chunks_created": len(points),
                "collection": "documents"
            }
            
        except Exception as e:
            logger.error(f"Ingestion error: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _extract_text(self, file_path: str, filename: str) -> str:
        text = ""
        
        try:
            if filename.lower().endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        except Exception as e:
                            logger.error(f"Error extracting page {page_num}: {e}")
            
            elif filename.lower().endswith(('.docx', '.doc')):
                doc = docx.Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs if para.text])
            
            elif filename.lower().endswith(('.txt', '.md')):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
            
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            raise
        
        return text
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        if not text:
            return []
            
        words = text.split()
        chunks = []
        
        if len(words) < chunk_size:
            return [text]
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk and len(chunk.strip()) > 10:
                chunks.append(chunk)
        
        return chunks
    
    async def search(self, request: RAGRequest) -> List[Dict[str, Any]]:
        if not self.qdrant:
            logger.warning("Qdrant not available, returning empty results")
            return []
            
        try:
            query_embedding = self.embedder.encode(request.query).tolist()
            
            results = self.qdrant.search(
                collection_name=request.collection,
                query_vector=query_embedding,
                limit=request.top_k,
                score_threshold=request.threshold
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.payload.get("text", ""),
                    "score": float(result.score),
                    "source": result.payload.get("source", "unknown"),
                    "chunk_index": result.payload.get("chunk_index", 0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {traceback.format_exc()}")
            return []

engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        engine = RAGEngine()
        await engine.connect_to_qdrant()
        logger.info("RAG Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}")
        # Don't raise - let service start but return unhealthy
        engine = RAGEngine()  # Create engine without qdrant connection

@app.get("/health")
async def health_check():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    qdrant_status = "disconnected"
    collections_count = 0
    
    if engine.qdrant:
        try:
            collections = engine.qdrant.get_collections()
            qdrant_status = "connected"
            collections_count = len(collections.collections)
        except Exception as e:
            logger.error(f"Health check Qdrant error: {e}")
            qdrant_status = "error"
    
    status = "healthy" if qdrant_status == "connected" else "degraded"
    
    return {
        "status": status,
        "qdrant_status": qdrant_status,
        "collections": collections_count,
        "embedding_model": EMBEDDING_MODEL,
        "device": engine.device
    }

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not engine.qdrant:
        raise HTTPException(status_code=503, detail="Qdrant service not available")
    
    try:
        result = await engine.ingest_file(file)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search")
async def search_documents(request: RAGRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        results = await engine.search(request)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "JARVIS RAG Service",
        "version": "1.0.0",
        "status": "online"
    }
