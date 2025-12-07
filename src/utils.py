from typing import List, Dict, Any
import json


def format_results(results: List[Dict[str, Any]], indent: int = 2) -> str:
    return json.dumps(results, indent=indent)


def calculate_statistics(chunks: List[Any]) -> Dict[str, Any]:
    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0
    sources = set(c.metadata.get('source', 'unknown') for c in chunks)
    
    return {
        'total_chunks': len(chunks),
        'total_characters': total_chars,
        'average_chunk_size': round(avg_chunk_size, 2),
        'unique_sources': len(sources),
        'sources': list(sources)
    }
