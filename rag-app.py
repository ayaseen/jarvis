# services/rag/app.py - COMPLETE ROBUST VERSION
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
    MilvusException
)
import PyPDF2
import docx
import logging
import torch
import uuid
import traceback
import asyncio
import time
import numpy as np
import json
import hashlib
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS RAG Service with Milvus", version="3.0.0")

# Configuration
MILVUS_HOST = os.getenv('MILVUS_HOST', 'milvus')
MILVUS_PORT = int(os.getenv('MILVUS_PORT', '19530'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
TRANSFORMERS_CACHE = os.getenv('TRANSFORMERS_CACHE', '/models/transformers')

# Model dimensions
MODEL_DIMENSIONS = {
    'sentence-transformers/all-MiniLM-L6-v2': 384,
    'sentence-transformers/all-MiniLM-L12-v2': 384,
    'sentence-transformers/all-mpnet-base-v2': 768,
    'BAAI/bge-base-en-v1.5': 768,
    'BAAI/bge-small-en-v1.5': 384,
}

EMBEDDING_DIM = 384

class RAGRequest(BaseModel):
    query: str
    collection: str = "documents"
    top_k: int = 5
    threshold: float = 0.3  # Lowered default threshold for better recall

class DocumentChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any]

class MilvusManager:
    def __init__(self, embedding_dim: int):
        self.connected = False
        self.collections = {}
        self.connection_alias = "default"
        self.embedding_dim = embedding_dim
        
    async def connect(self, max_retries: int = 30) -> bool:
        """Connect to Milvus with retry logic"""
        for attempt in range(max_retries):
            try:
                # Disconnect if already connected
                try:
                    connections.disconnect(self.connection_alias)
                except:
                    pass
                
                # Connect to Milvus
                connections.connect(
                    alias=self.connection_alias,
                    host=MILVUS_HOST,
                    port=MILVUS_PORT,
                    timeout=30
                )
                
                # Test connection
                server_info = utility.get_server_version()
                self.connected = True
                logger.info(f"Connected to Milvus {server_info} at {MILVUS_HOST}:{MILVUS_PORT}")
                
                # Initialize collections
                await self.init_collections()
                return True
                
            except Exception as e:
                logger.warning(f"Milvus connection attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt >= max_retries - 1:
                    logger.error("Failed to connect to Milvus after maximum retries")
                    return False
                await asyncio.sleep(2)
        
        return False
    
    async def init_collections(self):
        """Initialize default collections with proper schema"""
        collection_names = ["documents", "conversations", "knowledge"]
        
        for collection_name in collection_names:
            try:
                if utility.has_collection(collection_name):
                    # DO NOT DROP - Load existing collection
                    logger.info(f"Loading existing collection: {collection_name}")
                    collection = Collection(collection_name)
                    collection.load()
                    self.collections[collection_name] = collection
                    logger.info(f"Loaded collection {collection_name} with {collection.num_entities} entities")
                    continue
                
                # Define schema for collection
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=100,
                        description="Primary ID"
                    ),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.embedding_dim,
                        description="Text embedding vector"
                    ),
                    FieldSchema(
                        name="text",
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                        description="Original text content"
                    ),
                    FieldSchema(
                        name="source",
                        dtype=DataType.VARCHAR,
                        max_length=255,
                        description="Source document name"
                    ),
                    FieldSchema(
                        name="chunk_index",
                        dtype=DataType.INT64,
                        description="Chunk index in document"
                    ),
                    FieldSchema(
                        name="timestamp",
                        dtype=DataType.INT64,
                        description="Creation timestamp"
                    ),
                    FieldSchema(
                        name="metadata",
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                        description="Additional metadata as JSON"
                    )
                ]
                
                schema = CollectionSchema(
                    fields=fields,
                    description=f"{collection_name} collection for JARVIS RAG"
                )
                
                # Create collection
                collection = Collection(
                    name=collection_name,
                    schema=schema,
                    consistency_level="Strong"
                )
                
                # Create index for vector field - OPTIMIZED
                index_params = {
                    "metric_type": "COSINE",  # Using COSINE for normalized vectors
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                
                collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                
                # Load collection into memory
                collection.load()
                logger.info(f"Created and loaded collection: {collection_name}")
                
                self.collections[collection_name] = collection
                
            except Exception as e:
                logger.error(f"Error initializing collection {collection_name}: {e}")
                logger.error(traceback.format_exc())
    
    def get_collection(self, name: str) -> Optional[Collection]:
        """Get collection by name"""
        if name in self.collections:
            return self.collections[name]
        
        try:
            if utility.has_collection(name):
                collection = Collection(name)
                collection.load()
                self.collections[name] = collection
                return collection
        except Exception as e:
            logger.error(f"Error loading collection {name}: {e}")
        
        return None
    
    async def insert_vectors(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]],
        sources: List[str],
        chunk_indices: List[int],
        metadata_list: List[Dict[str, Any]] = None
    ) -> int:
        """Insert vectors into Milvus collection"""
        collection = self.get_collection(collection_name)
        if not collection:
            await self.init_collections()
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection {collection_name} not found and could not be created")
        
        # Prepare data
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        timestamp = int(time.time() * 1000)
        timestamps = [timestamp] * len(texts)
        
        # Convert metadata to JSON strings
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        metadata_strings = [json.dumps(m) for m in metadata_list]
        
        # Log sample data for debugging
        logger.info(f"Inserting {len(texts)} vectors into {collection_name}")
        logger.info(f"Sample text: {texts[0][:100] if texts else 'N/A'}...")
        logger.info(f"Sample source: {sources[0] if sources else 'N/A'}")
        
        # Prepare entities for insertion
        entities = [
            ids,
            embeddings,
            texts,
            sources,
            chunk_indices,
            timestamps,
            metadata_strings
        ]
        
        try:
            # Insert data
            result = collection.insert(entities)
            
            # Flush to ensure data is persisted
            collection.flush()
            
            logger.info(f"Successfully inserted {len(texts)} vectors into {collection_name}")
            
            # Verify insertion
            collection_count = collection.num_entities
            logger.info(f"Collection {collection_name} now has {collection_count} entities")
            
            return len(texts)
            
        except Exception as e:
            logger.error(f"Error inserting vectors: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def search_vectors(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Milvus with improved relevance"""
        collection = self.get_collection(collection_name)
        if not collection:
            logger.warning(f"Collection {collection_name} not found")
            return []
        
        try:
            # Log collection info
            logger.info(f"Searching in collection {collection_name} with {collection.num_entities} entities")
            
            # Normalize query embedding for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = (np.array(query_embedding) / query_norm).tolist()
            
            # Search parameters - OPTIMIZED
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}  # Increased for better recall
            }
            
            # Perform search with higher limit to filter later
            search_limit = min(top_k * 3, 50)  # Get more results for filtering
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=search_limit,
                output_fields=["text", "source", "chunk_index", "timestamp", "metadata"],
                consistency_level="Strong"
            )
            
            # Format and filter results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    score = float(hit.score)
                    
                    # Log high-scoring results for debugging
                    if score > 0.5:
                        logger.info(f"High relevance result: score={score:.3f}, text={hit.entity.get('text', '')[:100]}...")
                    
                    if score >= threshold:
                        # Parse metadata
                        metadata = {}
                        metadata_str = hit.entity.get("metadata", "{}")
                        try:
                            metadata = json.loads(metadata_str) if metadata_str else {}
                        except:
                            pass
                        
                        formatted_results.append({
                            "id": hit.id,
                            "text": hit.entity.get("text", ""),
                            "source": hit.entity.get("source", "unknown"),
                            "chunk_index": hit.entity.get("chunk_index", 0),
                            "score": score,
                            "timestamp": hit.entity.get("timestamp", 0),
                            "metadata": metadata
                        })
            
            # Sort by score and limit to top_k
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            formatted_results = formatted_results[:top_k]
            
            logger.info(f"Search returned {len(formatted_results)} results above threshold {threshold}")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            logger.error(traceback.format_exc())
            return []
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        collection = self.get_collection(collection_name)
        if not collection:
            return {"error": "Collection not found"}
        
        try:
            stats = {
                "name": collection_name,
                "num_entities": collection.num_entities,
                "loaded": True,
                "description": collection.description
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

class RAGEngine:
    def __init__(self):
        # Check if GPU should be used
        use_gpu = os.getenv('EMBEDDING_USE_GPU', 'true').lower() == 'true'
        
        if use_gpu and torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            if free_memory > 1e9:  # At least 1GB free
                self.device = 'cuda'
                logger.info(f"Using GPU for embeddings (free memory: {free_memory/1e9:.2f}GB)")
            else:
                self.device = 'cpu'
                logger.warning(f"Low GPU memory ({free_memory/1e9:.2f}GB), using CPU for embeddings")
        else:
            self.device = 'cpu'
            logger.info("Using CPU for embeddings")
        
        # Set cache directory
        os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
        os.environ['HF_HOME'] = TRANSFORMERS_CACHE
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = TRANSFORMERS_CACHE
        
        global EMBEDDING_DIM
        
        # Initialize embedding dimension
        self.embedding_dim = MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 384)
        
        try:
            self.embedder = SentenceTransformer(
                EMBEDDING_MODEL,
                device=self.device,
                cache_folder=TRANSFORMERS_CACHE
            )
            
            # Normalize embeddings for better cosine similarity
            self.embedder.normalize_embeddings = True
            
            # Get actual embedding dimension
            test_embedding = self.embedder.encode("test")
            actual_dim = len(test_embedding)
            self.embedding_dim = actual_dim
            
            if actual_dim != EMBEDDING_DIM:
                logger.info(f"Updating EMBEDDING_DIM from {EMBEDDING_DIM} to {actual_dim}")
                EMBEDDING_DIM = actual_dim
            
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL} on {self.device} (dim={self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Initialize Milvus manager
        self.milvus = MilvusManager(self.embedding_dim)
    
    async def connect(self) -> bool:
        """Connect to Milvus"""
        return await self.milvus.connect()
    
    def extract_text(self, file_path: str, filename: str) -> str:
        """Extract text from various file formats with improved parsing"""
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
            
            elif filename.lower().endswith(('.txt', '.md', '.json', '.csv', '.log', '.py', '.js', '.html', '.xml')):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
            
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            # Clean up text
            text = self.clean_text(text)
            logger.info(f"Extracted {len(text)} characters from {filename}")
                
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            raise
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        # Trim
        text = text.strip()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks with improved chunking"""
        if not text:
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        # Split by sentences for better context preservation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.strip()) > 10:
                        chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Keep last few sentences for overlap
                    overlap_sentences = current_chunk[-2:] if len(current_chunk) > 1 else current_chunk
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) > 10:
                chunks.append(chunk_text)
        
        logger.info(f"Created {len(chunks)} chunks with size ~{chunk_size} words")
        return chunks
    
    async def ingest_document(self, file: UploadFile, collection: str = "documents") -> Dict[str, Any]:
        """Ingest a document into Milvus with improved processing"""
        temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        
        try:
            # Save uploaded file
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Calculate file hash for deduplication
            with open(temp_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Extract text
            text = self.extract_text(temp_path, file.filename)
            
            if not text or len(text.strip()) < 10:
                raise ValueError("No text content extracted from file")
            
            logger.info(f"Processing document: {file.filename} ({len(text)} characters)")
            
            # Chunk text with smaller chunks for better precision
            chunks = self.chunk_text(text, chunk_size=300, overlap=50)
            logger.info(f"Created {len(chunks)} chunks from {file.filename}")
            
            # Log sample chunks for debugging
            if chunks:
                logger.info(f"First chunk: {chunks[0][:200]}...")
                if len(chunks) > 1:
                    logger.info(f"Last chunk: {chunks[-1][:200]}...")
            
            # Generate embeddings with normalization
            embeddings = []
            batch_size = 32
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_embeddings = self.embedder.encode(
                    batch, 
                    show_progress_bar=False,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                embeddings.extend(batch_embeddings.tolist())
            
            # Prepare metadata
            sources = [file.filename] * len(chunks)
            chunk_indices = list(range(len(chunks)))
            metadata_list = [
                {
                    "filename": file.filename,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "file_hash": file_hash,
                    "ingested_at": datetime.utcnow().isoformat()
                }
                for chunk in chunks
            ]
            
            # Insert into Milvus
            count = await self.milvus.insert_vectors(
                collection,
                chunks,
                embeddings,
                sources,
                chunk_indices,
                metadata_list
            )
            
            document_id = str(uuid.uuid4())
            
            # Verify insertion by searching for a sample
            if chunks:
                test_query = chunks[0][:50]  # Use first 50 chars as test
                test_embedding = self.embedder.encode(
                    test_query,
                    normalize_embeddings=True
                ).tolist()
                
                test_results = await self.milvus.search_vectors(
                    collection,
                    test_embedding,
                    top_k=1,
                    threshold=0.5
                )
                
                if test_results:
                    logger.info(f"Verification successful: Found ingested content with score {test_results[0]['score']:.3f}")
                else:
                    logger.warning("Verification warning: Could not find ingested content in search")
            
            return {
                "status": "success",
                "filename": file.filename,
                "chunks_created": count,
                "collection": collection,
                "document_id": document_id,
                "file_hash": file_hash,
                "text_length": len(text),
                "sample_text": text[:200] + "..." if len(text) > 200 else text
            }
            
        except Exception as e:
            logger.error(f"Document ingestion error: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def search(self, request: RAGRequest) -> Dict[str, Any]:
        """Search for similar documents with improved relevance"""
        try:
            logger.info(f"Searching for: '{request.query}' in collection '{request.collection}'")
            
            # Generate query embedding with normalization
            query_embedding = self.embedder.encode(
                request.query,
                normalize_embeddings=True
            ).tolist()
            
            # Search in Milvus with lower threshold for better recall
            results = await self.milvus.search_vectors(
                request.collection,
                query_embedding,
                request.top_k,
                request.threshold
            )
            
            # Log search results for debugging
            logger.info(f"Search returned {len(results)} results")
            if results:
                logger.info(f"Top result: score={results[0]['score']:.3f}, text={results[0]['text'][:100]}...")
            
            return {
                "results": results,
                "count": len(results),
                "query": request.query,
                "collection": request.collection
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            logger.error(traceback.format_exc())
            return {"results": [], "count": 0, "error": str(e)}

# Global engine instance
engine: Optional[RAGEngine] = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global engine
    try:
        engine = RAGEngine()
        connected = await engine.connect()
        
        if connected:
            logger.info("RAG Engine initialized successfully with Milvus")
        else:
            logger.warning("RAG Engine running but Milvus connection failed - will retry on requests")
            
    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}")
        logger.error(traceback.format_exc())
        engine = RAGEngine()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    if engine and engine.milvus.connected:
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except:
            pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_info = {
        "status": "unhealthy",
        "vector_db": "milvus",
        "milvus_status": "disconnected",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "device": "unknown",
        "collections": []
    }
    
    if engine:
        health_info["device"] = engine.device
        health_info["embedding_dim"] = engine.embedding_dim
        
        # Try to reconnect if not connected
        if not engine.milvus.connected:
            await engine.connect()
        
        if engine.milvus.connected:
            try:
                # Get Milvus server info
                server_version = utility.get_server_version()
                collections = utility.list_collections()
                
                health_info["milvus_status"] = "connected"
                health_info["milvus_version"] = server_version
                health_info["collections"] = collections
                health_info["status"] = "healthy"
                
                # Get collection details
                collection_details = []
                for coll_name in collections:
                    stats = await engine.milvus.get_collection_stats(coll_name)
                    collection_details.append(stats)
                health_info["collection_details"] = collection_details
                
            except Exception as e:
                health_info["milvus_status"] = f"error: {str(e)}"
                health_info["status"] = "degraded"
    
    return health_info

@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    collection: str = Query(default="documents")
):
    """Upload and ingest a document"""
    if not engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")
    
    # Try to reconnect if not connected
    if not engine.milvus.connected:
        connected = await engine.connect()
        if not connected:
            raise HTTPException(status_code=503, detail="Cannot connect to Milvus")
    
    result = await engine.ingest_document(file, collection)
    return result

@app.post("/search")
async def search_documents(request: RAGRequest):
    """Search for similar documents"""
    if not engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")
    
    # Try to reconnect if not connected
    if not engine.milvus.connected:
        connected = await engine.connect()
        if not connected:
            return {"results": [], "count": 0, "error": "Milvus not connected"}
    
    result = await engine.search(request)
    return result

@app.get("/collections")
async def list_collections():
    """List available collections"""
    if not engine:
        return {"collections": [], "error": "Engine not initialized"}
    
    if not engine.milvus.connected:
        await engine.connect()
    
    if not engine.milvus.connected:
        return {"collections": [], "error": "Milvus not connected"}
    
    try:
        collections = utility.list_collections()
        collection_details = []
        
        for coll_name in collections:
            stats = await engine.milvus.get_collection_stats(coll_name)
            collection_details.append(stats)
        
        return {
            "collections": collections,
            "details": collection_details,
            "total": len(collections)
        }
    except Exception as e:
        return {"collections": [], "error": str(e)}

@app.delete("/collections/{collection_name}/clear")
async def clear_collection(collection_name: str):
    """Clear all data in a collection but keep the collection"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not engine.milvus.connected:
        await engine.connect()
    
    if not engine.milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus not connected")
    
    try:
        collection = engine.milvus.get_collection(collection_name)
        if collection:
            # Delete all entities
            expr = "id != ''"  # This matches all entities
            collection.delete(expr)
            collection.flush()
            
            return {
                "status": "success",
                "message": f"Collection {collection_name} cleared",
                "remaining_entities": collection.num_entities
            }
        else:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/sample")
async def sample_collection(
    collection_name: str,
    limit: int = Query(default=5)
):
    """Get sample documents from a collection"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not engine.milvus.connected:
        await engine.connect()
    
    if not engine.milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus not connected")
    
    try:
        collection = engine.milvus.get_collection(collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")
        
        # Query to get sample entities
        results = collection.query(
            expr="id != ''",  # Get all
            output_fields=["text", "source", "chunk_index"],
            limit=limit
        )
        
        return {
            "collection": collection_name,
            "samples": results,
            "total": collection.num_entities
        }
    except Exception as e:
        logger.error(f"Error sampling collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "JARVIS RAG Service",
        "version": "3.0.0",
        "vector_db": "Milvus",
        "status": "online",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIM if engine else MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 384),
        "milvus_endpoint": f"{MILVUS_HOST}:{MILVUS_PORT}",
        "device": engine.device if engine else "unknown",
        "endpoints": {
            "health": "/health",
            "ingest": "/ingest",
            "search": "/search",
            "collections": "/collections",
            "docs": "/docs"
        }
    }
