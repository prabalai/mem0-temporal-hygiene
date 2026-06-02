#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Hygiene and De-duplication script for Mem0 OSS + Qdrant.
Walks through the Mem0 Qdrant collection, groups semantically similar facts,
and uses LLM-driven consolidation to resolve contradictions and remove obsolete points.
"""

import os
import sys
import json
import urllib.request
import logging
from datetime import datetime, timezone
import numpy as np
import dotenv
import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memory-hygiene")

# Load environment variables
dotenv.load_dotenv(os.path.expanduser("~/.hermes/.env"))

# Ensure paths
sys.path.insert(0, "/root/.hermes")
sys.path.insert(0, "/root/.hermes/plugins")
sys.path.insert(0, "/opt/hermes-agent")

from mem0 import Memory

def load_mem0_config() -> dict:
    default_cfg = {
        "llm_model": "hermes-nvidia-fast",
        "llm_base_url": "http://localhost:20130/v1",
        "llm_api_key": os.environ.get("OPENAI_API_KEY", "***"),
        "embedder_provider": "openai",
        "embedder_model": "nvidia/nv-embedqa-e5-v5",
        "embedder_base_url": "http://localhost:20130/v1",
        "embedder_api_key": os.environ.get("OPENAI_API_KEY", "***"),
        "embedder_model_kwargs": {"extra_body": {"input_type": "passage"}, "omit_dimensions": True},
        "embedding_dims": 1024,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "collection_name": "hermes_memories",
        "user_id": "default_user",
    }
    
    path = os.path.expanduser("~/.hermes/mem0_oss.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                # Resolve masking
                for k, v in user_cfg.items():
                    if v == "***":
                        if "llm" in k or "embedder" in k:
                            user_cfg[k] = os.environ.get("OPENAI_API_KEY", "***")
                default_cfg.update(user_cfg)
        except Exception as e:
            logger.warning("Failed to load mem0_oss.json: %s", e)
            
    return default_cfg

def build_mem0_params(cfg: dict) -> dict:
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": cfg["llm_model"],
                "openai_base_url": cfg["llm_base_url"],
                "api_key": cfg["llm_api_key"],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": cfg.get("embedder_provider", "openai"),
            "config": {
                "model": cfg["embedder_model"],
                "openai_base_url": cfg.get("embedder_base_url", cfg["llm_base_url"]),
                "api_key": cfg.get("embedder_api_key", cfg["llm_api_key"]),
                "embedding_dims": cfg.get("embedding_dims"),
                "model_kwargs": cfg.get("embedder_model_kwargs", {}),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": cfg["qdrant_host"],
                "port": cfg["qdrant_port"],
                "collection_name": cfg["collection_name"],
                "embedding_model_dims": cfg["embedding_dims"],
            },
        },
        "version": "v1.1",
    }

def cosine_similarity(v1, v2) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def fetch_qdrant_points(cfg: dict) -> list:
    url = f"http://{cfg['qdrant_host']}:{cfg['qdrant_port']}/collections/{cfg['collection_name']}/points/scroll"
    payload = {
        "limit": 500,
        "with_payload": True,
        "with_vector": True
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        return res.get("result", {}).get("points", [])

def group_points_by_similarity(points: list, threshold: float = 0.82) -> list:
    groups = []
    visited = set()
    
    for i, p_i in enumerate(points):
        if p_i["id"] in visited:
            continue
            
        vector_i = p_i.get("vector", {}).get("")
        if not vector_i:
            continue
            
        current_group = [p_i]
        visited.add(p_i["id"])
        
        for j in range(i + 1, len(points)):
            p_j = points[j]
            if p_j["id"] in visited:
                continue
                
            vector_j = p_j.get("vector", {}).get("")
            if not vector_j:
                continue
                
            sim = cosine_similarity(vector_i, vector_j)
            if sim >= threshold:
                current_group.append(p_j)
                visited.add(p_j["id"])
                
        if len(current_group) >= 2:
            groups.append(current_group)
            
    return groups

def analyze_group_with_llm(client: openai.OpenAI, model_name: str, group: list) -> dict:
    memories_str = []
    for p in group:
        p_id = p["id"]
        p_payload = p.get("payload", {})
        p_data = p_payload.get("data") or p_payload.get("memory") or ""
        created_at = p_payload.get("created_at") or ""
        memories_str.append({
            "id": p_id,
            "text": p_data,
            "created_at": created_at
        })
        
    prompt = f"""You are a Database Memory Janitor. You are given a group of vector-similar memories stored about a user.
They may represent duplicates, overlapping details, or contradictory facts where one fact is outdated and was replaced by a newer decision.

Review this list of memories:
{json.dumps(memories_str, indent=2, ensure_ascii=False)}

Your Goal:
1. Determine if there are contradictory or overlapping facts.
2. If there are contradictions, the newer facts (based on 'created_at' timestamp) override older ones.
3. If facts are fully duplicated or represent minor redundant details, merge them into a single, clean, concise memory (using Russian, keeping key facts like paths, names, numbers intact).
4. If some memories are obsolete or fully redundant, output their IDs in "deletions".
5. If an existing memory needs to be updated with consolidated clean text, output its ID and consolidated text in "updates".
6. If the memories are actually about completely different things and should both be kept as they are, return empty updates and deletions.

Format the output strictly as a JSON object, like this:
{{
  "deletions": ["uuid-to-delete-1", "uuid-to-delete-2"],
  "updates": [
    {{
      "id": "uuid-to-keep-and-update",
      "text": "New consolidated fact text goes here."
    }}
  ]
}}

No extra comments, no markdown code blocks, just raw JSON.
"""
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a database cleanup utility. Respond ONLY with valid JSON structure."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = res.choices[0].message.content.strip()
        # Clean markdown wrapper if any
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                content = "\n".join(lines[1:-1])
        return json.loads(content)
    except Exception as e:
        logger.error("LLM group analysis failed: %s", e)
        return {"deletions": [], "updates": []}

def main():
    logger.info("Initializing Memory Hygiene Job...")
    cfg = load_mem0_config()
    
    # Initialize Mem0 Memory SDK
    mem0_params = build_mem0_params(cfg)
    m = Memory.from_config(mem0_params)
    
    # Initialize OpenAI Client via OmniRoute
    client = openai.OpenAI(
        base_url=cfg["llm_base_url"],
        api_key=cfg["llm_api_key"]
    )
    model_name = cfg["llm_model"]
    
    # Fetch all points
    try:
        points = fetch_qdrant_points(cfg)
        logger.info(f"Successfully fetched {len(points)} points from Qdrant.")
    except Exception as e:
        logger.error(f"Failed to fetch points from Qdrant: {e}")
        sys.exit(1)
        
    if not points:
        logger.info("Database is empty. Nothing to clean.")
        return
        
    # Group by similarity
    logger.info("Grouping semantically similar points...")
    groups = group_points_by_similarity(points, threshold=0.82)
    logger.info(f"Found {len(groups)} groups of potentially similar/duplicate memories.")
    
    total_deletions = 0
    total_updates = 0
    
    for idx, group in enumerate(groups):
        logger.info(f"Analyzing group {idx+1}/{len(groups)} with {len(group)} items...")
        for p in group:
            p_payload = p.get("payload", {})
            p_data = p_payload.get("data") or p_payload.get("memory") or ""
            logger.info(f"  - [{p['id'][:8]}] {p_payload.get('created_at', '')[:10]}: {p_data[:80]}...")
            
        decision = analyze_group_with_llm(client, model_name, group)
        logger.info(f"  LLM Decision: {json.dumps(decision, ensure_ascii=False)}")
        
        deletions = decision.get("deletions", [])
        updates = decision.get("updates", [])
        
        # Performance updates via Mem0 SDK to sync vector embedding changes
        for update_item in updates:
            up_id = update_item.get("id")
            up_text = update_item.get("text")
            if up_id and up_text:
                try:
                    m.update(up_id, up_text)
                    logger.info(f"  [UPDATED] Fact {up_id[:8]} -> '{up_text}'")
                    total_updates += 1
                except Exception as e:
                    logger.error(f"  Failed to update fact {up_id}: {e}")
                    
        # Perform deletions via Mem0 SDK
        for del_id in deletions:
            try:
                m.delete(del_id)
                logger.info(f"  [DELETED] Fact {del_id[:8]}")
                total_deletions += 1
            except Exception as e:
                logger.error(f"  Failed to delete fact {del_id}: {e}")
                
    logger.info(f"Hygiene job complete. Deletions check: deleted {total_deletions} points, updated {total_updates} points.")

if __name__ == "__main__":
    main()
