#!/usr/bin/env python3
# debug_milvus.py - Run this to see what's actually in your Milvus database

from pymilvus import connections, Collection, utility
import json

def debug_milvus():
    print("=" * 80)
    print("MILVUS DEBUG REPORT")
    print("=" * 80)
    
    try:
        # Connect to Milvus
        connections.connect(host='localhost', port=19530)
        print("✓ Connected to Milvus\n")
        
        # List all collections
        collections = utility.list_collections()
        print(f"Available Collections: {collections}\n")
        
        # Check each collection
        for coll_name in collections:
            print(f"\n{'='*40}")
            print(f"Collection: {coll_name}")
            print(f"{'='*40}")
            
            collection = Collection(coll_name)
            collection.load()
            
            num_entities = collection.num_entities
            print(f"Number of entities: {num_entities}")
            
            if num_entities > 0:
                # Get ALL entities to see what's there
                results = collection.query(
                    expr="id != ''",
                    output_fields=["text", "source", "chunk_index", "metadata"],
                    limit=min(100, num_entities)  # Get up to 100 entities
                )
                
                print(f"\nShowing {len(results)} entities:\n")
                
                # Group by source
                sources = {}
                for i, doc in enumerate(results):
                    source = doc.get('source', 'unknown')
                    if source not in sources:
                        sources[source] = []
                    sources[source].append(doc)
                
                # Display by source
                for source, docs in sources.items():
                    print(f"\n📄 Source: {source}")
                    print(f"   Chunks: {len(docs)}")
                    
                    for j, doc in enumerate(docs[:3]):  # Show first 3 chunks per source
                        text = doc.get('text', '')
                        chunk_idx = doc.get('chunk_index', 0)
                        metadata = doc.get('metadata', '{}')
                        
                        # Parse metadata
                        try:
                            meta_dict = json.loads(metadata) if metadata else {}
                        except:
                            meta_dict = {}
                        
                        print(f"\n   Chunk {chunk_idx}:")
                        print(f"   Text preview: {text[:200]}...")
                        print(f"   Metadata: {meta_dict}")
                    
                    if len(docs) > 3:
                        print(f"   ... and {len(docs) - 3} more chunks")
            else:
                print("   ⚠️  Collection is empty!")
        
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        
        # Summary statistics
        total_entities = 0
        for coll_name in collections:
            collection = Collection(coll_name)
            total_entities += collection.num_entities
            print(f"{coll_name}: {collection.num_entities} entities")
        
        print(f"\nTotal entities across all collections: {total_entities}")
        
        if total_entities < 10:
            print("\n⚠️  WARNING: Very few entities found!")
            print("This suggests documents are not being properly chunked and indexed.")
            print("\nPossible issues:")
            print("1. Document chunking is failing")
            print("2. Embeddings are not being generated")
            print("3. Milvus insertion is failing")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_milvus()
