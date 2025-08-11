#!/usr/bin/env python3
# test_rag.py - Test RAG upload and search directly

import requests
import json
import time

def test_rag_system():
    RAG_URL = "http://localhost:8002"
    ORCHESTRATOR_URL = "http://localhost:8000"
    
    print("=" * 80)
    print("TESTING RAG SYSTEM")
    print("=" * 80)
    
    # 1. Check RAG health
    print("\n1. Checking RAG health...")
    try:
        response = requests.get(f"{RAG_URL}/health")
        health = response.json()
        print(f"   RAG Status: {health.get('status')}")
        print(f"   Milvus: {health.get('milvus_status')}")
        print(f"   Device: {health.get('device')}")
        print(f"   Model: {health.get('embedding_model')}")
    except Exception as e:
        print(f"   ❌ RAG health check failed: {e}")
        return
    
    # 2. Create test document about Amjad Yaseen
    print("\n2. Creating test document...")
    test_content = """
    Amjad Yaseen is a Senior Solution Architect at Red Hat.
    He has been working at Red Hat for almost 4 years.
    Amjad specializes in cloud architecture and Kubernetes.
    He is an expert in OpenShift and containerization technologies.
    Amjad helps enterprises adopt cloud-native solutions.
    He works with customers to design scalable infrastructure.
    Amjad is based in Saudi Arabia and works with Middle East clients.
    He has extensive experience in DevOps practices and automation.
    """
    
    # Save as file
    with open('/tmp/amjad_test.txt', 'w') as f:
        f.write(test_content)
    
    # 3. Upload directly to RAG service
    print("\n3. Uploading to RAG service...")
    try:
        with open('/tmp/amjad_test.txt', 'rb') as f:
            files = {'file': ('amjad_yaseen.txt', f, 'text/plain')}
            response = requests.post(
                f"{RAG_URL}/ingest",
                files=files,
                params={'collection': 'documents'}
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Upload successful!")
            print(f"   Chunks created: {result.get('chunks_created')}")
            print(f"   Text length: {result.get('text_length')}")
            print(f"   Sample: {result.get('sample_text', '')[:100]}...")
        else:
            print(f"   ❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return
    except Exception as e:
        print(f"   ❌ Upload error: {e}")
        return
    
    # 4. Wait for indexing
    print("\n4. Waiting for indexing...")
    time.sleep(2)
    
    # 5. Check collection stats
    print("\n5. Checking collection stats...")
    try:
        response = requests.get(f"{RAG_URL}/collections")
        collections = response.json()
        for detail in collections.get('details', []):
            if detail['name'] == 'documents':
                print(f"   Documents collection: {detail['num_entities']} entities")
    except Exception as e:
        print(f"   ❌ Collections check failed: {e}")
    
    # 6. Test search directly on RAG
    print("\n6. Testing RAG search...")
    test_queries = [
        "Who is Amjad Yaseen?",
        "Amjad Yaseen",
        "Red Hat architect",
        "architect Red Hat years"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            response = requests.post(
                f"{RAG_URL}/search",
                json={
                    "query": query,
                    "collection": "documents",
                    "top_k": 3,
                    "threshold": 0.2  # Very low threshold
                }
            )
            
            if response.status_code == 200:
                results = response.json()
                print(f"   Found {results['count']} results")
                
                for i, result in enumerate(results.get('results', [])[:2]):
                    print(f"\n   Result {i+1}:")
                    print(f"   Score: {result['score']:.3f}")
                    print(f"   Source: {result['source']}")
                    print(f"   Text: {result['text'][:150]}...")
            else:
                print(f"   ❌ Search failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Search error: {e}")
    
    # 7. Test through orchestrator
    print("\n\n7. Testing through orchestrator/chat...")
    try:
        response = requests.post(
            f"{ORCHESTRATOR_URL}/api/chat",
            json={
                "message": "Who is Amjad Yaseen?",
                "use_rag": True,
                "collection": "documents"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Context used: {result.get('context_used')}")
            print(f"   Context count: {result.get('context_count')}")
            print(f"   Response: {result.get('response', '')[:200]}...")
        else:
            print(f"   ❌ Chat failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Chat error: {e}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_rag_system()
