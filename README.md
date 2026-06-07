# mem0-temporal-hygiene

**Temporal context, conservative write-time conflict detection, soft-deletes, guarded deterministic merges, and hash-based caching for Mem0 OSS + Qdrant setups.**

---

### ⚡ Why Mem0 Temporal Hygiene? (30-Second Overview)

Standard Mem0 OSS stores user preferences and agent memories as flat, coordinate-sparse vectors. Over time, this leads to two major breakdowns:
1. **Time Blindness:** If a user says *"I like dark mode"* in May, and *"I prefer light mode"* in June, both facts remain in the database. Vector similarity cannot differentiate between them, leading to contradictory contexts.
2. **Polar/Toggle Collapse:** Embeddings for polar opposites (e.g., *"turn ON notifications"* and *"turn OFF notifications"*) are semantically close (cosine similarity ~0.85). Naive deduplication merges them, corrupting the configuration.

This project implements a **Metadata Validity Overlay** + **Periodic Hygiene Engine** to solve these issues.

| Feature / Problem | Standard Mem0 OSS | With Temporal Hygiene Overlay |
| :--- | :--- | :--- |
| **Updating Preferences** | Aggressively mutates or duplicates | Marks old as `superseded`, links to `winner_id` |
| **Opposite Toggles** | High risk of false-positive merge | Negation & toggle guards prevent wrong merges |
| **Auditing History** | Hard delete (records lost) | Soft delete (`status: deleted`), full audit log intact |
| **Decay & Recency** | Static vector weights | Dynamic temporal decay calculated at retrieval |
| **API Token Costs** | Merges everything with LLM weekly | Guards skip unchanged groups using hash cache |

---

### 🏗️ Memory Pipeline Architecture

```text
[ User Prompt / Tool Call ]
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              1. CONSERVATIVE WRITE GUARD                │
│  - Prefetch similar items (limit=5)                     │
│  - Run Negation/Toggle analysis                         │
│  - Classify relation (related / conflict / duplicate)   │
│  - Append metadata: provenance, trust_tier, confidence  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               2. METADATA VALIDITY LAYER                │
│  - Write payload with `status: active`                  │
│  - Soft-delete via `status: deleted` (keeps history)    │
│  - Mark replaced memories `status: superseded`          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 3. DB-LEVEL RETRIEVAL                   │
│  - Qdrant payload filters automatically exclude         │
│    superseded and deleted records                       │
│  - Temporal Decay applied (relevance = sim * decay)     │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Core Features

### 1. Trust Tiers and Key Extraction
Before writing, the engine deterministic extracts keys (e.g., `theme`, `editor`) and matches them. Facts are labeled with **Trust Tiers**:
- `user_explicit`: Stated directly by the user (highest priority, `confidence: 1.0`).
- `agent_decision`: Deduced by the agent during tasks (`confidence: 0.8`).
- `tool_log`: Extracted from errors or scripts (`confidence: 0.5`).

### 2. Time-Aware Validity & Soft-Deletes
We never physically delete your vectors. Suppanted preferences get updated with `status: "superseded"`, `superseded_by: "<id>"`. Deleted items get `status: "deleted"`. Retrieval filters at Qdrant-level ignore them, keeping the context clean.

### 3. Negation & Toggle Guards
If a new memory has a similarity of `0.92` to an existing one but contains antonyms or negation patterns (e.g., `not`, `don't`, `disable`, `false`, `on/off`), auto-merge is blocked. The records are linked as `suspected_conflict` instead.

### 4. Hash-Based Cluster Cache
The periodic hygiene script hashes semantic clusters by content and point IDs. If a cluster has not changed, the engine reuses the previous merge decision, cutting down LLM API token consumption by up to 90%.

---

## 🚀 Quick Start in 60 Seconds

### 1. Installation

Copy the enhanced plugin and hygiene script into your Hermes setup:

```bash
# 1. Update the Mem0 OSS plugin
cp plugin/__init__.py ~/.hermes/plugins/mem0-oss/__init__.py

# 2. Add the hygiene script
cp scripts/memory-hygiene.py ~/.hermes/scripts/memory-hygiene.py
chmod +x ~/.hermes/scripts/memory-hygiene.py

# 3. Add the audit shell script
cp scripts/memory-hygiene-audit.sh ~/.hermes/scripts/memory-hygiene-audit.sh
chmod +x ~/.hermes/scripts/memory-hygiene-audit.sh

# 4. Restart the gateway
systemctl restart hermes-gateway
```

### 2. Safe Execution (Audit / Dry-Run)

Validate proposed deduplications and merges without making any writes to your database:

```bash
python3 ~/.hermes/scripts/memory-hygiene.py --dry-run
```

Alternatively, use the cron-friendly wrapper that stays silent unless anomalies are detected:

```bash
~/.hermes/scripts/memory-hygiene-audit.sh
```

---

## 📝 Configuration

The engine reads your existing `~/.hermes/mem0_oss.json` configuration config. Fill in your LLM and Qdrant details:

```json
{
  "llm_model": "your-model-name",
  "llm_base_url": "http://localhost:20130/v1",
  "llm_api_key": "your-key",
  "embedder_provider": "openai",
  "embedder_model": "text-embedding-3-small",
  "embedder_base_url": "http://localhost:20130/v1",
  "embedding_dims": 1536,
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "collection_name": "hermes_default",
  "user_id": "default"
}
```

---

## 📜 Scholarly References & Context

- **Mem0 Issue [#4896](https://github.com/mem0ai/mem0/issues/4896):** Community discussions on handling contradiction in memory stores.
- **Generative Agents (Park et al., 2023):** Groundwork research on building LLM agents with memory retrieval weights: `recency × importance × relevance`.
- **MemoryBank (Zhong et al., 2023):** Practical strategies for modeling memory decay rates based on usage.

## 📄 License

MIT. Free for custom integrations and self-hosted environments. Contributions welcome!
