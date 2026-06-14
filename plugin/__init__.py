"""Mem0 OSS memory plugin — self-hosted memory via local embedder + Qdrant.

Unlike the bundled mem0 plugin (which uses MemoryClient for Mem0 Cloud),
this one wraps the local ``mem0.Memory`` class so everything stays on your
machine: embeddings computed locally (bge-m3), vectors in Qdrant, LLM for
fact extraction via OmniRoute.

Config loaded from $HERMES_HOME/mem0_oss.json (optional). Defaults suit
Dmitry's setup: OmniRoute on :20130, Qdrant on :6333, bge-m3 embedder.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


def _default_config() -> dict:
    return {
        "llm_model": "AgentCron",
        "llm_base_url": "http://localhost:20130/v1",
        "llm_api_key": os.environ.get("OPENAI_API_KEY", "sk-hermes-admin"),
        "embedder_provider": "openai",
        "embedder_model": "nvidia/nv-embedqa-e5-v5",
        "embedder_base_url": "http://localhost:20130/v1",
        "embedder_api_key": os.environ.get("OPENAI_API_KEY", "sk-hermes-admin"),
        "embedder_model_kwargs": {"extra_body": {"input_type": "passage"}},
        "embedding_dims": None,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "collection_name": "hermes_dmitry",
        "user_id": "dmitry",
    }


def _load_config() -> dict:
    from hermes_constants import get_hermes_home
    cfg = _default_config()
    path = get_hermes_home() / "mem0_oss.json"
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("mem0_oss.json parse error: %s", e)
    return cfg


def _clip_text(text: str, limit: int) -> str:
    text = text or ""
    if limit and len(text) > limit:
        return text[:limit] + "…"
    return text



MEMORY_SCHEMA_VERSION = "2026-06-11-mfs-write-guard-v2"
NEGATION_RE = re.compile(
    r"\b(не|нет|никогда|запрет|запрещ|не надо|no|not|never|disable|disabled|off|without|avoid|don't|do not)\b",
    re.IGNORECASE,
)
TOGGLE_RE = re.compile(
    r"\b(enable|enabled|disable|disabled|on|off|turn on|turn off|включи|включать|выключи|выключать|активир|деактивир)\b",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _extract_subject_keys(text: str) -> set[str]:
    t = (text or "").lower()
    entities = [
        "omniroute", "qdrant", "plex", "torrserver", "transmission", "rclone", 
        "yandex", "backup", "cookie", "vps", "dreame", "obsidian", "sasha", 
        "nadya", "nestor", "andrey", "stt", "tts", "whisper", "slack", 
        "telegram", "ha", "home assistant", "фото", "видео", "мантра", 
        "песня", "санскрит", "порт", "таймаут", "логи"
    ]
    found = {ent for ent in entities if ent in t}
    # Match structural variables like 'name = value' or 'name: value'
    match = re.search(r"^\s*([a-zа-я0-9_\-\s]{3,30})\s*[:=]", t)
    if match:
        var_name = match.group(1).strip()
        # Only add if it's a solid word combination
        if len(var_name.split()) <= 3:
            found.add(var_name)
    return found


def _calculate_temporal_decay(created_at_str: str, source: str) -> float:
    try:
        if not created_at_str:
            return 1.0
        # Parse ISO client time or created_at
        dt_str = created_at_str.replace("Z", "+00:00")
        # Handle formats like 2026-06-06T18:00:58
        if "T" in dt_str and "+" not in dt_str and "-" not in dt_str[10:]:
            dt_str += "+00:00"
        created_at = datetime.fromisoformat(dt_str)
        now = datetime.now(timezone.utc)
        days = (now - created_at).days
        if days < 0:
            days = 0
            
        # Select lambda based on source
        # user explicit never decays
        if "explicit" in (source or "") or (source or "") in ("user", "user_explicit"):
            decay_rate = 0.0
        elif (source or "") in ("tool", "tool_log"):
            decay_rate = 0.05  # fast decay: half-life ~14 days
        else:
            decay_rate = 0.005 # default gentle decay (half-life ~138 days)
            
        import math
        return math.exp(-decay_rate * days)
    except Exception:
        return 1.0


def _ensure_optimized_collection(cfg: dict) -> None:
    import urllib.request
    import urllib.error
    import json
    
    collection_name = cfg.get("collection_name")
    if not collection_name:
        return
        
    host = cfg.get("qdrant_host", "localhost")
    port = cfg.get("qdrant_port", 6333)
    url = f"http://{host}:{port}/collections/{collection_name}"
    
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            # Collection exists
            return
    except urllib.error.HTTPError as e:
        if e.code != 404:
            logger.warning("Qdrant get collection error: %s", e)
            return
    except Exception as e:
        logger.warning("Qdrant connection wrapper error: %s", e)
        return
        
    # Collection does not exist, create it with optimized settings
    logger.info("Pre-creating optimized Qdrant collection: %s", collection_name)
    dims = cfg.get("embedding_dims") or 1024
    create_body = {
        "vectors": {
            "size": dims,
            "distance": "Cosine",
            "on_disk": True
        },
        "hnsw_config": {
            "on_disk": True
        },
        "quantization_config": {
            "scalar": {
                "type": "int8",
                "quantile": 0.99,
                "always_ram": True
            }
        }
    }
    create_body["sparse_vectors"] = {
        "bm25": {
            "modifier": "idf"
        }
    }
    
    try:
        req_create = urllib.request.Request(
            url,
            data=json.dumps(create_body).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="PUT"
        )
        with urllib.request.urlopen(req_create, timeout=5) as resp:
            logger.info("Pre-created collection response: %s", resp.read().decode('utf-8').strip())
    except Exception as e:
        logger.error("Failed to pre-create optimized Qdrant collection %s: %s", collection_name, e)


def _fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def _has_negation_or_toggle(text: str) -> bool:
    return bool(NEGATION_RE.search(text or "") or TOGGLE_RE.search(text or ""))


def _memory_text(item: dict) -> str:
    return item.get("memory") or item.get("text") or item.get("data") or ""

def _build_mem0_config(cfg: dict) -> dict:
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


PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Получить все факты о пользователе из долгосрочной памяти (Mem0 OSS self-hosted). "
        "Вызывай в начале разговора чтобы понять кто это."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mfs_path": {"type": "string", "description": "Ограничить выборку конкретным путем MFS (например, /user/preferences/music/)."}
        },
        "required": []
    },
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Семантический поиск по памяти. Возвращает релевантные факты. "
        "Используй когда нужно вспомнить что-то специфичное о пользователе."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Что искать."},
            "top_k": {"type": "integer", "description": "Сколько результатов (default 5, max 20)."},
            "mfs_path": {"type": "string", "description": "Ограничить поиск конкретным путем MFS (например, /user/preferences/music/)."}
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "mem0_remember",
    "description": (
        "Сохранить факт о пользователе в долгосрочную память. Mem0 сам извлечёт "
        "атомарные факты и дедуплицирует. Используй при явных предпочтениях, "
        "корректировках, важных деталях."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Текст для запоминания."},
        },
        "required": ["content"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "Удалить факт о пользователе из долгосрочной памяти (Mem0 OSS) по его UUID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "UUID факта (отображается в квадратных скобках [ID: ...] в памяти)."}
        },
        "required": ["memory_id"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": (
        "Обновить существующий факт о пользователе в долгосрочной памяти по его UUID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "UUID факта для редактирования."},
            "content": {"type": "string", "description": "Новый текст факта, заменяющий старый."}
        },
        "required": ["memory_id", "content"],
    },
}


class Mem0OSSProvider(MemoryProvider):
    """Self-hosted Mem0 via local embedder + Qdrant + OmniRoute LLM."""

    def __init__(self):
        self._cfg: dict | None = None
        self._memory = None
        self._lock = threading.Lock()
        self._user_id = "dmitry"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None

    @property
    def name(self) -> str:
        return "mem0-oss"

    def is_available(self) -> bool:
        # Qdrant up?
        try:
            import urllib.request
            cfg = _load_config()
            url = f"http://{cfg['qdrant_host']}:{cfg['qdrant_port']}/"
            urllib.request.urlopen(url, timeout=2)
            import mem0  # noqa: F401
            return True
        except Exception:
            return False

    def _get_memory(self):
        with self._lock:
            if self._memory is not None:
                return self._memory
            from mem0 import Memory
            self._cfg = _load_config()
            _ensure_optimized_collection(self._cfg)
            self._memory = Memory.from_config(_build_mem0_config(self._cfg))
            return self._memory

    # -- MFS path taxonomy: ordered (most-specific first) rule table --
    _MFS_RULES: list[tuple[list[str], str]] = [
        # /user/preferences/*
        (["music", "jazz", "lo-fi", "песн", "джаз", "музык", "петь", "плейлист", "playlist", "spotify"], "/user/preferences/music/"),
        (["food", "еда", "кухня", "рецепт", "готов", "вегетар", "cuisine", "recipe", "cook"], "/user/preferences/food/"),
        (["lang", "язык", "locale", "локал", "english", "русск"], "/user/preferences/language/"),
        (["theme", "dark mode", "light mode", "тема", "оформлен", "шрифт", "font"], "/user/preferences/ui/"),
        (["hates", "prefers", "style", "любит", "не любит", "предпочитает", "стиль", "привычк"], "/user/preferences/style/"),
        # /user/health/
        (["здоров", "health", "лекарств", "медиц", "врач", "doctor", "аллерг", "allerg", "диагноз", "diagnosis"], "/user/health/"),
        # /user/schedule/
        (["wake", "sleep", "schedule", "утра", "вечера", "подъем", "график", "расписание", "routine", "рутин"], "/user/schedule/"),
        # /user/contacts/
        (["contact", "контакт", "телефон", "phone", "email", "почта", "адрес", "address"], "/user/contacts/"),
        # /family/*
        (["семь", "family", "ребён", "дет", "child", "жена", "wife", "муж", "husband", "родител", "parent"], "/family/"),
        # /projects/*
        (["проект", "project", "repo", "репо", "github", "gitlab", "sprint", "спринт", "задач", "task", "канбан", "kanban"], "/projects/"),
        # /system/config/* (specific services first)
        (["omniroute"], "/system/config/omniroute/"),
        (["qdrant"], "/system/config/qdrant/"),
        (["transmission"], "/system/config/transmission/"),
        (["plex"], "/system/config/plex/"),
        (["torrserver"], "/system/config/torrserver/"),
        (["hermes"], "/system/config/hermes/"),
        (["home assistant", " ha ", "homeassistant", "хоум ассистант"], "/system/config/homeassistant/"),
        (["telegram", "тг ", "tg "], "/system/config/telegram/"),
        (["obsidian"], "/system/config/obsidian/"),
        (["nginx", "caddy", "ssl", "cert", "сертификат", "domain", "домен", "dns", "webdav"], "/system/network/"),
        (["port", "key", "url", "порт", "ключ", "сеть", "token", "токен", "api_key", "secret"], "/system/config/"),
        # /system/backup/
        (["backup", "бэкап", "rclone", "rsync", "snapshot", "снэпшот", "yandex disk", "яндекс диск"], "/system/backup/"),
        # /system/cron/
        (["cron", "крон", "schedule job", "таймер", "systemd timer"], "/system/cron/"),
        # /system/media/
        (["фото", "photo", "видео", "video", "медиа", "media", "галере", "gallery", "альбом", "album"], "/system/media/"),
        # /education/
        (["учёб", "учеб", "школ", "school", "курс", "course", "урок", "lesson", "домашн", "homework", "экзамен", "exam"], "/education/"),
        # /finance/
        (["деньг", "money", "финанс", "finance", "бюджет", "budget", "расход", "expense", "доход", "income", "счёт", "invoice"], "/finance/"),
    ]

    def _extract_mfs_path(self, content: str) -> str:
        """Extract VFS/MFS path from fact content using rules first, LLM fallback."""
        text = (content or "").strip()
        if not text:
            return "/general/"

        t_low = text.lower()

        # 1. Rule-based fast path: scan ordered taxonomy
        for keywords, path in self._MFS_RULES:
            if any(kw in t_low for kw in keywords):
                return path

        # 2. LLM call fallback
        try:
            url = f"{self._cfg['llm_base_url']}/chat/completions"
            headers = {
                "Content-Type": "application/json",
            }
            if self._cfg.get('llm_api_key') and self._cfg['llm_api_key'] != "***":
                headers["Authorization"] = f"Bearer {self._cfg['llm_api_key']}"

            body = {
                "model": self._cfg["llm_model"],
                "messages": [
                    {
                        "role": "system", 
                        "content": (
                            "You are a VFS (Virtual File System) path extractor. Classify the input text into a path "
                            "resembling a folder structure (e.g., /user/preferences/music/, /projects/config/). "
                            "Output ONLY the single absolute path, lowercase, with trailing slash. Do not include markdown formatting or extra words."
                        )
                    },
                    {"role": "user", "content": "User listens to jazz in the morning"},
                    {"role": "assistant", "content": "/user/preferences/music/"},
                    {"role": "user", "content": "Dmitry hates dry updates"},
                    {"role": "assistant", "content": "/user/preferences/style/"},
                    {"role": "user", "content": "The backup cron script runs at 3 AM"},
                    {"role": "assistant", "content": "/system/backup/"},
                    {"role": "user", "content": "OmniRoute port is 20130"},
                    {"role": "assistant", "content": "/system/config/omniroute/"},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.0,
                "max_tokens": 50,
                "stream": False
            }

            import urllib.request
            req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                raw_path = res['choices'][0]['message']['content'].strip()
                
                # Sanitize the output
                path = raw_path.lower()
                path = re.sub(r"[`'\"#\s]", "", path)
                path = re.sub(r"^(path|vfspath|vfs|mfs):", "", path)
                if not path.startswith("/"):
                    path = "/" + path
                if not path.endswith("/"):
                    path = path + "/"
                path = re.sub(r"/+", "/", path)
                return path
        except Exception as e:
            logger.warning("LLM _extract_mfs_path failed: %s, falling back to /general/", e)
            return "/general/"

    def _get_active_filters(self, mfs_path: str = None, *, prefix_match: bool = False) -> dict:
        """Build Qdrant payload filters.
        
        When prefix_match=True, the mfs_path acts as a VFS directory prefix
        (e.g. /system/config/ matches /system/config/omniroute/ too).
        Qdrant doesn't support native prefix, so we pass the exact path and 
        handle prefix filtering in Python post-search.
        """
        filters = {
            "user_id": self._user_id,
            "NOT": [
                {"status": "superseded"},
                {"status": "deleted"}
            ]
        }
        if mfs_path and not prefix_match:
            filters["mfs_path"] = mfs_path
        # For prefix_match we filter in Python after retrieval
        return filters

    @staticmethod
    def _mfs_paths_related(path_a: str, path_b: str) -> bool:
        """Check if two MFS paths share a namespace (one is prefix/ancestor of the other).
        
        /system/config/ is related to /system/config/omniroute/
        /system/config/omniroute/ is related to /system/config/omniroute/
        /user/preferences/ is NOT related to /system/config/
        """
        if not path_a or not path_b:
            return True  # unknown path → conservative: assume related
        a = path_a.rstrip("/") + "/"
        b = path_b.rstrip("/") + "/"
        return a.startswith(b) or b.startswith(a)

    @staticmethod
    def _mfs_paths_same_namespace(path_a: str, path_b: str) -> bool:
        """Check if two MFS paths are in the SAME leaf namespace (exact match or siblings).
        
        /system/config/omniroute/ == /system/config/omniroute/ → True
        /system/config/ vs /system/config/omniroute/ → True (parent contains child)
        /system/config/omniroute/ vs /system/config/qdrant/ → False (different siblings)
        /user/preferences/music/ vs /system/config/ → False
        """
        if not path_a or not path_b:
            return True  # unknown → conservative
        a = path_a.rstrip("/") + "/"
        b = path_b.rstrip("/") + "/"
        return a == b or a.startswith(b) or b.startswith(a)

    def _base_metadata(self, *, source: str = "explicit_tool", confidence: float = 0.85) -> dict:
        now = _utc_now()
        return {
            "status": "active",
            "user_id": self._user_id,
            "memory_schema_version": MEMORY_SCHEMA_VERSION,
            "source": source,
            "provenance": source,
            "confidence": confidence,
            "source_confidence": confidence,
            "created_at_client": now,
            "updated_at_client": now,
        }

    def _search_write_candidates(self, m, content: str, *, mfs_path: str = None, top_k: int = 5) -> list[dict]:
        """Search for existing facts that might conflict with a new write.
        
        When mfs_path is provided, we search WITHOUT path filter in Qdrant 
        (to catch cross-namespace near-duplicates), then post-filter by MFS
        namespace relatedness. This is the VFS-aware write guard.
        """
        try:
            clipped = (content or "")[:800]
            if not clipped.strip():
                return []
            # Search broadly (no mfs_path filter) to catch cross-namespace conflicts
            res = m.search(query=clipped, filters=self._get_active_filters(), top_k=top_k * 2)
            items = res.get("results", []) if isinstance(res, dict) else res
            results = [r for r in items if _memory_text(r)]
            
            if mfs_path and results:
                # Annotate each candidate with its mfs_path relatedness
                for r in results:
                    cand_meta = r.get("metadata", {}) or {}
                    cand_path = cand_meta.get("mfs_path", "")
                    r["_mfs_related"] = self._mfs_paths_related(mfs_path, cand_path)
                    r["_mfs_same_ns"] = self._mfs_paths_same_namespace(mfs_path, cand_path)
                    r["_mfs_path"] = cand_path
                
                # Sort: same namespace first, then related, then unrelated
                results.sort(key=lambda x: (not x.get("_mfs_same_ns"), not x.get("_mfs_related")))
            
            return results[:top_k]
        except TypeError:
            res = m.search(query=(content or "")[:800], filters=self._get_active_filters(), limit=top_k * 2)
            items = res.get("results", []) if isinstance(res, dict) else res
            return [r for r in items if _memory_text(r)][:top_k]
        except Exception as e:
            logger.debug("mem0 write-candidate search failed: %s", e)
            return []

    def _classify_write_relation(self, content: str, candidates: list[dict]) -> dict:
        related = []
        possible_duplicates = []
        possible_conflicts = []
        new_guarded = _has_negation_or_toggle(content)
        keys_content = _extract_subject_keys(content)
        
        for c in candidates:
            cid = c.get("id") or c.get("memory_id") or ""
            if not cid:
                continue
            score = float(c.get("score") or 0.0)
            text = _memory_text(c)
            
            # Key/subject intersection safety check
            keys_cand = _extract_subject_keys(text)
            common_entities = keys_content.intersection(keys_cand)
            if common_entities:
                diff_content = keys_content - keys_cand
                diff_cand = keys_cand - keys_content
                specifiers = {"порт", "port", "timeout", "таймаут", "логи", "log", "backup", "бэкап", "почта", "email", "url", "путь", "path", "token", "токен"}
                has_diff_spec = bool(diff_content.intersection(specifiers) and diff_cand.intersection(specifiers))
                if has_diff_spec:
                    # They speak about different keys of the same entity (e.g. port vs timeout). Bypass classification.
                    logger.debug("Bypass: candidate has mismatching specifier keys compared to request")
                    continue

            if score >= 0.72:
                related.append(cid)
            if score >= 0.92 and not (new_guarded or _has_negation_or_toggle(text)):
                possible_duplicates.append(cid)
            if score >= 0.78 and (new_guarded or _has_negation_or_toggle(text)):
                possible_conflicts.append(cid)

        status = "none"
        if possible_conflicts:
            status = "suspected_conflict"
        elif possible_duplicates:
            status = "possible_duplicate"

        return {
            "conflict_status": status,
            "conflicts_with": possible_conflicts[:5],
            "possible_duplicate_of": possible_duplicates[:5],
            "related_memory_ids": related[:5],
            "write_guard": "detect_append_no_delete",
            "candidate_count": len(candidates),
        }

    def save_config(self, values, hermes_home):
        from pathlib import Path
        path = Path(hermes_home) / "mem0_oss.json"
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                pass
        existing.update({k: v for k, v in values.items() if v})
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    def get_config_schema(self):
        return [
            {"key": "user_id", "description": "User identifier", "default": "dmitry"},
            {"key": "llm_model", "description": "LLM via OmniRoute", "default": "AgentCron"},
            {"key": "llm_base_url", "description": "OmniRoute URL", "default": "http://localhost:20130/v1"},
            {"key": "embedder_model", "description": "Local embedding model", "default": "BAAI/bge-m3"},
            {"key": "qdrant_host", "description": "Qdrant host", "default": "localhost"},
            {"key": "qdrant_port", "description": "Qdrant port", "default": 6333},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._cfg = _load_config()
        self._user_id = kwargs.get("user_id") or self._cfg.get("user_id", "dmitry")

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 OSS Memory (self-hosted)\n"
            f"Активна. User: {self._user_id}. Хранилище: Qdrant локально. "
            "Эмбеддинги через OmniRoute/NVIDIA, LLM для извлечения фактов через OmniRoute.\n"
            "Инструменты: mem0_profile (всё о пользователе), mem0_search (поиск), "
            "mem0_remember (сохранить факт), mem0_update (обновить факт), mem0_delete (удалить факт)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return f"## Mem0 OSS Memory\n{result}" if result else ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _run():
            try:
                m = self._get_memory()
                # Clip search query to prevent embedding token limit errors (e.g. 512 tokens for NVIDIA NIM)
                clipped = query[:800] if query else ""
                res = m.search(query=clipped, filters=self._get_active_filters(), limit=5)
                items = res.get("results", []) if isinstance(res, dict) else res
                lines = []
                for r in items:
                    mem = r.get("memory", "")
                    if not mem:
                        continue
                    created_at = r.get("created_at") or ""
                    date_prefix = f"[{created_at[:10]}] " if created_at and len(created_at) >= 10 else ""
                    mem_id = r.get("id") or ""
                    id_suffix = f" [ID: {mem_id}]" if mem_id else ""
                    lines.append(f"- {date_prefix}{mem}{id_suffix}")
                with self._prefetch_lock:
                    self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug("mem0-oss prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0oss-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Disabled: auto-sync creates too much noise (session logs, not durable facts).
        # Memory is now write-only via explicit mem0_remember calls.
        # To re-enable, uncomment the block below.
        pass
        # def _sync():
        #     try:
        #         m = self._get_memory()
        #         user_limit = int((self._cfg or {}).get("sync_user_char_limit", 700))
        #         assistant_limit = int((self._cfg or {}).get("sync_assistant_char_limit", 700))
        #         messages = [
        #             {"role": "user", "content": _clip_text(user_content, user_limit)},
        #             {"role": "assistant", "content": _clip_text(assistant_content, assistant_limit)},
        #         ]
        #         m.add(messages, user_id=self._user_id)
        #     except Exception as e:
        #         logger.warning("mem0-oss sync failed: %s", e)
        #
        # if self._sync_thread and self._sync_thread.is_alive():
        #     self._sync_thread.join(timeout=5.0)
        # self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0oss-sync")
        # self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, REMEMBER_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        try:
            m = self._get_memory()
        except Exception as e:
            return tool_error(f"Mem0 OSS unavailable: {e}")

        if tool_name == "mem0_profile":
            try:
                mfs_path = args.get("mfs_path")
                res = m.get_all(filters=self._get_active_filters(mfs_path))
                items = res.get("results", []) if isinstance(res, dict) else res
                if not items:
                    return json.dumps({"result": "Память пуста."})
                lines = []
                for r in items:
                    mem = r.get("memory", "")
                    if not mem:
                        continue
                    created_at = r.get("created_at") or ""
                    date_prefix = f"[{created_at[:10]}] " if created_at and len(created_at) >= 10 else ""
                    metadata = r.get("metadata", {}) or {}
                    path_str = f" ({metadata.get('mfs_path')})" if metadata.get("mfs_path") else ""
                    mem_id = r.get("id") or ""
                    id_suffix = f" [ID: {mem_id}]" if mem_id else ""
                    lines.append(f"{date_prefix}{mem}{path_str}{id_suffix}")
                return json.dumps({"result": "\n".join(lines), "count": len(lines)}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Ошибка получения профиля: {e}")

        if tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Нужен параметр query")
            top_k = min(int(args.get("top_k", 5)), 20)
            mfs_path = args.get("mfs_path")
            try:
                # Clip search query to prevent embedding token limit errors (e.g. 512 tokens for NVIDIA NIM)
                clipped = query[:800]
                # Query a wider pool of candidates to allow decay re-ranking
                search_limit = min(top_k * 3, 50)
                res = m.search(query=clipped, filters=self._get_active_filters(mfs_path), limit=search_limit)
                items = res.get("results", []) if isinstance(res, dict) else res
                if not items:
                    return json.dumps({"result": "Релевантных фактов не найдено."}, ensure_ascii=False)
                
                # Perform hybrid scoring (re-ranking)
                scored_items = []
                for r in items:
                    mem = r.get("memory", "")
                    if not mem:
                        continue
                    
                    metadata = r.get("metadata", {}) or {}
                    created_at = r.get("created_at") or metadata.get("created_at") or metadata.get("created_at_client") or ""
                    source = metadata.get("source") or metadata.get("provenance") or "agent_decision"
                    confidence = float(metadata.get("confidence") or metadata.get("source_confidence") or 0.85)
                    
                    decay_factor = _calculate_temporal_decay(created_at, source)
                    sim_score = float(r.get("score", 0.0))
                    
                    # Hybrid formula: 70% similarity, 30% decay and confidence
                    final_score = 0.7 * sim_score + 0.3 * decay_factor * confidence
                    
                    scored_items.append({
                        "id": r.get("id") or "",
                        "memory": mem,
                        "created_at": created_at,
                        "raw_score": sim_score,
                        "score": final_score,
                        "decay_factor": decay_factor,
                        "confidence": confidence,
                        "mfs_path": metadata.get("mfs_path")
                    })
                
                # Sort descending by updated final score
                scored_items.sort(key=lambda x: x["score"], reverse=True)
                
                # Slice down to requested top_k
                final_results = scored_items[:top_k]
                
                out = []
                for r in final_results:
                    created_at = r["created_at"]
                    date_prefix = f"[{created_at[:10]}] " if created_at and len(created_at) >= 10 else ""
                    path_str = f" ({r['mfs_path']})" if r.get('mfs_path') else ""
                    out.append({
                        "id": r["id"],
                        "memory": f"{date_prefix}{r['memory']}{path_str}",
                        "score": r["score"],
                        "raw_score": r["raw_score"],
                        "created_at": created_at
                    })
                return json.dumps({"results": out, "count": len(out)}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Поиск не удался: {e}")

        if tool_name == "mem0_remember":
            content = args.get("content", "")
            if not content:
                return tool_error("Нужен параметр content")
            try:
                limit = int((self._cfg or {}).get("remember_char_limit", 900))
                clipped_content = _clip_text(content, limit)
                mfs_path = self._extract_mfs_path(clipped_content)
                candidates = self._search_write_candidates(m, clipped_content, mfs_path=mfs_path)
                relation = self._classify_write_relation(clipped_content, candidates)
                metadata = self._base_metadata(source="explicit_mem0_remember", confidence=0.85)
                metadata.update(relation)
                metadata["content_fingerprint"] = _fingerprint(clipped_content)
                metadata["mfs_path"] = mfs_path

                # Conservative write path: detect and annotate conflicts/duplicates, but let Mem0
                # keep its normal extraction UX. We do not soft-delete or supersede anything here.
                res = m.add(
                    clipped_content,
                    user_id=self._user_id,
                    metadata=metadata,
                    infer=bool((self._cfg or {}).get("remember_infer", True)),
                )
                items = res.get("results", []) if isinstance(res, dict) else res
                added = [r for r in items if r.get("event") == "ADD"]
                return json.dumps({
                    "result": f"Сохранено {len(added)} факт(ов).",
                    "facts": [r.get("memory", "") for r in added],
                    "mfs_path": mfs_path,
                    "write_guard": {
                        "conflict_status": relation["conflict_status"],
                        "conflicts_with": relation["conflicts_with"],
                        "possible_duplicate_of": relation["possible_duplicate_of"],
                    },
                }, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Сохранение не удалось: {e}")

        if tool_name == "mem0_delete":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Нужен параметр memory_id")
            try:
                # Get the existing memory text to preserve it during soft-delete update
                mem_item = m.get(memory_id)
                if not mem_item:
                    return tool_error(f"Факт с ID {memory_id} не найден.")
                content = mem_item.get("memory") or mem_item.get("text") or ""
                # Perform soft-delete by setting status payload to "deleted" while keeping provenance.
                metadata = self._base_metadata(source="explicit_mem0_delete", confidence=1.0)
                metadata.update({"status": "deleted", "deleted_at_client": _utc_now()})
                m.update(memory_id, content, metadata=metadata)
                return json.dumps({"result": f"Факт {memory_id} успешно помечен как удалённый (soft-delete)."}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Не удалось удалить факт: {e}")

        if tool_name == "mem0_update":
            memory_id = args.get("memory_id", "")
            content = args.get("content", "")
            if not memory_id or not content:
                return tool_error("Нужны параметры memory_id и content")
            try:
                mfs_path = self._extract_mfs_path(content)
                candidates = [c for c in self._search_write_candidates(m, content, mfs_path=mfs_path) if (c.get("id") or c.get("memory_id")) != memory_id]
                relation = self._classify_write_relation(content, candidates)
                metadata = self._base_metadata(source="explicit_mem0_update", confidence=0.95)
                metadata.update(relation)
                metadata["content_fingerprint"] = _fingerprint(content)
                metadata["updated_memory_id"] = memory_id
                metadata["mfs_path"] = mfs_path
                m.update(memory_id, content, metadata=metadata)
                return json.dumps({
                    "result": f"Факт {memory_id} успешно обновлён.",
                    "mfs_path": mfs_path,
                    "write_guard": {
                        "conflict_status": relation["conflict_status"],
                        "conflicts_with": relation["conflicts_with"],
                        "possible_duplicate_of": relation["possible_duplicate_of"],
                    },
                }, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Не удалось обновить факт: {e}")

        return tool_error(f"Неизвестный tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)


def register(ctx) -> None:
    ctx.register_memory_provider(Mem0OSSProvider())
