# mem0-temporal-hygiene

**Temporal context, CRUD tools, and automated deduplication for Mem0 OSS + Qdrant setups.**

> Mem0 OSS stores user facts as flat vectors in Qdrant. Over time, contradictory and duplicate facts accumulate — the system has no built-in mechanism to resolve conflicts or expire outdated information. This plugin and maintenance script solve that problem.

## The Problem

When using [Mem0 OSS](https://github.com/mem0ai/mem0) with a local Qdrant vector database, semantic memory suffers from three fundamental issues:

1. **Time-Blindness** — Vector similarity search returns facts sorted by cosine similarity, not by recency. A fact from May 15 and a contradicting fact from June 2 look "synchronous" to the model. The agent cannot tell which one is current.

2. **Weak Conflict Resolution** — Mem0's `.add()` pipeline uses LLM-driven extraction with MD5 hash deduplication. This catches byte-for-byte duplicates but misses semantic contradictions (e.g., "use path A" vs. "path A was replaced by path B"). Both facts coexist permanently. The Mem0 team has [explicitly closed](https://github.com/mem0ai/mem0/issues/4896) this as "not planned" — their v3 architecture treats it as by-design behavior.

3. **No Agent-Side CRUD** — Standard Mem0 plugins expose only `search`, `profile`, and `remember` tools. When the agent discovers an outdated fact mid-conversation, it has no way to delete or update it without writing raw HTTP calls to Qdrant's REST API.

### Real-World Examples

- **Ghost folder loop**: An old fact containing a path with an invisible Zero-Width Joiner character gets retrieved on every project scan. The agent copies the corrupted path and recreates a phantom folder, even though a newer fact says "don't use ZWJ in paths."
- **Stale recovery instructions**: When a service fails, the agent retrieves a month-old recovery fact ("ask the user to re-authenticate in the browser") instead of the current fix ("the real issue is AppArmor blocking the SOCKS proxy").

## The Solution

This repository provides two components:

### 1. Enhanced Mem0 OSS Plugin (`__init__.py`)

A drop-in replacement for the standard `mem0-oss` Hermes Agent plugin that adds:

- **Temporal context in all outputs** — Every fact returned by `prefetch`, `mem0_profile`, and `mem0_search` now includes its creation date and UUID:
  ```
  - [2026-06-02] AppArmor blocks Chromium via SOCKS proxy, use --no-proxy-server [ID: 9da2f37a-...]
  - [2026-05-15] To recover cookies, log into Chromium browser [ID: ab12cd34-...]
  ```
  The LLM can now see that the June fact supersedes the May fact.

- **`mem0_update(memory_id, content)`** — Update an existing fact's text and re-embed it in Qdrant.
- **`mem0_delete(memory_id)`** — Delete an outdated or incorrect fact by its UUID.

### 2. Memory Hygiene Script (`memory-hygiene.py`)

An automated maintenance script that:

1. Fetches all vectors from the Qdrant collection
2. Groups points by cosine similarity (threshold ≥ 0.82)
3. For each group, sends the facts to an LLM with instructions to:
   - Identify contradictions (newer date wins)
   - Merge duplicates into a single consolidated fact
   - Flag obsolete entries for deletion
4. Applies updates and deletions via the Mem0 SDK

Designed to run weekly as a cron job or systemd timer.

## Installation

### Prerequisites

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) with the `mem0-oss` plugin
- Qdrant running locally (default: `localhost:6333`)
- Mem0 OSS Python package (`pip install mem0ai`)
- An OpenAI-compatible LLM endpoint (e.g., OmniRoute, vLLM, ollama)

### Plugin Installation

```bash
# Back up your existing plugin
cp -r ~/.hermes/plugins/mem0-oss ~/.hermes/plugins/mem0-oss.bak

# Copy the enhanced plugin
cp plugin/__init__.py ~/.hermes/plugins/mem0-oss/__init__.py

# Restart your Hermes gateway to pick up the new tools
systemctl restart hermes-gateway
```

### Hygiene Script Installation

```bash
# Copy the script
cp scripts/memory-hygiene.py ~/.hermes/scripts/memory-hygiene.py
chmod +x ~/.hermes/scripts/memory-hygiene.py

# Test it (dry run logs to stdout)
python3 ~/.hermes/scripts/memory-hygiene.py
```

### Scheduling (Optional)

Add to your weekly maintenance cron or systemd timer:

```bash
# Example: run every Sunday at 2 AM
echo "0 2 * * 0 root /usr/bin/python3 /root/.hermes/scripts/memory-hygiene.py >> /var/log/memory-hygiene.log 2>&1" \
  > /etc/cron.d/memory-hygiene
```

Or integrate into an existing Hermes cron maintenance script.

## Configuration

The plugin reads its configuration from `~/.hermes/mem0_oss.json`:

```json
{
  "llm_model": "your-model-name",
  "llm_base_url": "http://localhost:20130/v1",
  "llm_api_key": "your-api-key",
  "embedder_provider": "openai",
  "embedder_model": "nvidia/nv-embedqa-e5-v5",
  "embedder_base_url": "http://localhost:20130/v1",
  "embedder_api_key": "your-api-key",
  "embedding_dims": 1024,
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "collection_name": "hermes_user",
  "user_id": "your_user_id"
}
```

## How It Works

### Temporal Context Flow

```
Before (standard mem0-oss):
  prefetch → "- User prefers dark mode"
                (no date, no ID, no way to know if this is current)

After (this plugin):
  prefetch → "- [2026-05-15] User prefers dark mode [ID: abc123]"
  prefetch → "- [2026-06-01] User switched to light mode [ID: def456]"
                (LLM sees dates, picks the June fact, can delete the May fact)
```

### Hygiene Script Flow

```
1. Fetch all 90 vectors from Qdrant
2. Compute pairwise cosine similarity
3. Group vectors with similarity ≥ 0.82
   → Found 8 groups of potentially conflicting facts

4. For each group, LLM produces a decision:
   {
     "deletions": ["old-uuid-1", "old-uuid-2"],
     "updates": [{"id": "keep-uuid", "text": "Consolidated fact text"}]
   }

5. Apply via Mem0 SDK: m.update() and m.delete()
   → Result: 8 deletions, 6 updates (90 → 82 clean facts)
```

## Background & Related Work

- **Mem0 Issues**: [#4896](https://github.com/mem0ai/mem0/issues/4896), [#4904](https://github.com/mem0ai/mem0/issues/4904) — Community reports of the same problem; closed as "not planned" by maintainers.
- **Generative Agents** (Park et al., 2023): Introduced `recency × importance × relevance` scoring for memory retrieval — the theoretical foundation for temporal prioritization.
- **MemoryBank** (Zhong et al., 2023): Ebbinghaus forgetting curve applied to LLM memory — memories decay over time unless reinforced by access.
- **RecallM** (2023): Updatable long-term memory with belief updating when contradictory information arrives.

## License

MIT

## Contributing

Issues and PRs welcome. This started as a self-hosted fix for a real production problem — if you're running Mem0 OSS with Qdrant at scale, your contributions will help the community.
