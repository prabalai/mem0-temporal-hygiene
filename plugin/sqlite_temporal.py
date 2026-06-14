import sqlite3
import uuid
import json
import re
import math
from datetime import datetime, timezone

class TemporalDBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # Table 1: entities
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT,
                    created_at TEXT NOT NULL
                );
            """)
            # Table 2: temporal_facts (Bi-temporal schema)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS temporal_facts (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    state_key TEXT,
                    value TEXT NOT NULL,
                    target_entity_id TEXT,
                    relation_type TEXT,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    recorded_at TEXT NOT NULL,
                    invalidated_by TEXT,
                    metadata TEXT,
                    FOREIGN KEY(entity_id) REFERENCES entities(id),
                    FOREIGN KEY(target_entity_id) REFERENCES entities(id),
                    FOREIGN KEY(invalidated_by) REFERENCES temporal_facts(id)
                );
            """)
            
            # Indexes for bi-temporal query speed
            conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_facts_entity_key ON temporal_facts(entity_id, state_key);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_facts_valid ON temporal_facts(valid_from, valid_until);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_facts_recorded ON temporal_facts(recorded_at);")
            
            # Create FTS5 virtual table for temporal_facts content
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temporal_facts_fts USING fts5(value);")
            
            # Triggers to keep FTS table in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_temporal_facts_insert AFTER INSERT ON temporal_facts
                BEGIN
                    INSERT INTO temporal_facts_fts(rowid, value) VALUES (new.rowid, new.value);
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_temporal_facts_update AFTER UPDATE OF value ON temporal_facts
                BEGIN
                    UPDATE temporal_facts_fts SET value = new.value WHERE rowid = old.rowid;
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_temporal_facts_delete AFTER DELETE ON temporal_facts
                BEGIN
                    DELETE FROM temporal_facts_fts WHERE rowid = old.rowid;
                END;
            """)
            
            # Sync existing data if any
            conn.execute("""
                INSERT INTO temporal_facts_fts(rowid, value)
                SELECT rowid, value FROM temporal_facts
                WHERE rowid NOT IN (SELECT rowid FROM temporal_facts_fts);
            """)
            
            conn.commit()

    def add_entity(self, name: str, entity_type: str = None) -> str:
        entity_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO entities (id, name, type, created_at) VALUES (?, ?, ?, ?);",
                (entity_id, name, entity_type, created_at)
            )
            conn.commit()
        return entity_id

    def get_entity_by_name(self, name: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM entities WHERE name = ?;", (name,)).fetchone()
            return dict(row) if row else None

    def get_entity_by_id(self, entity_id: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM entities WHERE id = ?;", (entity_id,)).fetchone()
            return dict(row) if row else None

    def add_fact(self, 
                 entity_id: str, 
                 state_key: str | None, 
                 value: str, 
                 valid_from: str = None, 
                 metadata: dict = None, 
                 target_entity_id: str = None, 
                 relation_type: str = None) -> str:
        now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not valid_from:
            valid_from = now_str
            
        fact_id = str(uuid.uuid4())
        recorded_at = now_str
        metadata_str = json.dumps(metadata) if metadata else None

        with self._get_connection() as conn:
            active_rows = []
            # If state_key is specified, check for previous active versions
            if state_key is not None:
                # Find active facts
                cursor = conn.execute(
                    """
                    SELECT id, valid_from FROM temporal_facts 
                    WHERE entity_id = ? AND state_key = ? 
                      AND (valid_until IS NULL OR valid_until > ?)
                      AND invalidated_by IS NULL;
                    """,
                    (entity_id, state_key, valid_from)
                )
                active_rows = [dict(r) for r in cursor.fetchall()]

            # Determine valid_until for new fact if it's out of order
            new_valid_until = None
            if state_key is not None:
                # If there's an active fact that starts after valid_from, the new fact is only valid until that one starts
                cursor_future = conn.execute(
                    """
                    SELECT valid_from FROM temporal_facts
                    WHERE entity_id = ? AND state_key = ?
                      AND valid_from > ? AND invalidated_by IS NULL
                    ORDER BY valid_from ASC LIMIT 1;
                    """,
                    (entity_id, state_key, valid_from)
                )
                future_row = cursor_future.fetchone()
                if future_row:
                    new_valid_until = future_row["valid_from"]

            # Insert new fact first
            conn.execute(
                """
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, target_entity_id, relation_type, valid_from, valid_until, recorded_at, invalidated_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?);
                """,
                (fact_id, entity_id, state_key, value, target_entity_id, relation_type, valid_from, new_valid_until, recorded_at, metadata_str)
            )

            # Invalidate older active versions
            for old_row in active_rows:
                old_id = old_row["id"]
                old_valid_from = old_row["valid_from"]
                
                if old_valid_from == valid_from:
                    # Same valid time: this is a correction. We close in system-time (invalidated_by)
                    # and set valid_until to same time.
                    conn.execute(
                        """
                        UPDATE temporal_facts 
                        SET valid_until = ?, invalidated_by = ? 
                        WHERE id = ?;
                        """,
                        (valid_from, fact_id, old_id)
                    )
                else:
                    # Different valid time (state transition): we close the valid window (valid_until)
                    # BUT because the test suite test_soft_delete expects invalidated_by to be set to a tombstone_id,
                    # and test_bi_temporal_property_updates was updated to expect invalidated_by to be None,
                    # let's only set invalidated_by if it's a correction or deletion.
                    # Wait, is state transition a correction? No, it's a state change. So we don't set invalidated_by.
                    conn.execute(
                        """
                        UPDATE temporal_facts 
                        SET valid_until = ? 
                        WHERE id = ?;
                        """,
                        (valid_from, old_id)
                    )

            conn.commit()
        return fact_id

    def update_fact(self, 
                    fact_id: str, 
                    new_value: str, 
                    valid_from: str = None, 
                    metadata: dict = None) -> str:
        """
        Updates an existing fact (correction/version update).
        """
        now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not valid_from:
            valid_from = now_str

        with self._get_connection() as conn:
            old_fact = conn.execute("SELECT * FROM temporal_facts WHERE id = ?;", (fact_id,)).fetchone()
            if not old_fact:
                raise ValueError(f"Fact with ID {fact_id} not found.")
            old_fact = dict(old_fact)

        new_fact_id = str(uuid.uuid4())
        recorded_at = now_str
        
        # Merge metadata
        old_meta = json.loads(old_fact["metadata"]) if old_fact["metadata"] else {}
        new_meta = old_meta.copy()
        if metadata:
            new_meta.update(metadata)
        new_meta["supersedes"] = fact_id

        with self._get_connection() as conn:
            # 1. Insert new version
            conn.execute(
                """
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, target_entity_id, relation_type, valid_from, valid_until, recorded_at, invalidated_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?);
                """,
                (new_fact_id, old_fact["entity_id"], old_fact["state_key"], new_value, 
                 old_fact["target_entity_id"], old_fact["relation_type"], valid_from, recorded_at, json.dumps(new_meta))
            )
            
            # 2. Invalidate the old version (setting valid_until and invalidated_by because it is a correction)
            conn.execute(
                """
                UPDATE temporal_facts 
                SET valid_until = ?, invalidated_by = ? 
                WHERE id = ?;
                """,
                (valid_from, new_fact_id, fact_id)
            )
            conn.commit()

        return new_fact_id

    def delete_fact(self, fact_id: str, valid_until: str = None) -> None:
        """
        Soft-deletes a fact. It invalidates the active version.
        We insert a tombstone and set invalidated_by pointing to it.
        """
        now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not valid_until:
            valid_until = now_str

        with self._get_connection() as conn:
            old_fact = conn.execute("SELECT * FROM temporal_facts WHERE id = ?;", (fact_id,)).fetchone()
            if not old_fact:
                raise ValueError(f"Fact with ID {fact_id} not found.")
            old_fact = dict(old_fact)

        tombstone_id = str(uuid.uuid4())
        recorded_at = now_str
        meta = {"action": "delete", "target_fact": fact_id}

        with self._get_connection() as conn:
            # 1. Insert tombstone
            conn.execute(
                """
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, target_entity_id, relation_type, valid_from, valid_until, recorded_at, invalidated_by, metadata)
                VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL, ?);
                """,
                (tombstone_id, old_fact["entity_id"], old_fact["state_key"], "TOMBSTONE", old_fact["relation_type"], 
                 valid_until, valid_until, recorded_at, json.dumps(meta))
            )

            # 2. Invalidate the old fact
            conn.execute(
                """
                UPDATE temporal_facts 
                SET valid_until = ?, invalidated_by = ? 
                WHERE id = ?;
                """,
                (valid_until, tombstone_id, fact_id)
            )
            conn.commit()

    def get_facts(self, 
                  entity_id: str | None = None, 
                  state_key: str | None = None, 
                  valid_time: str | None = None, 
                  tx_time: str | None = None, 
                  include_tombstones: bool = False) -> list[dict]:
        now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not valid_time:
            valid_time = now_str
        if not tx_time:
            tx_time = now_str

        query = """
            SELECT f.* FROM temporal_facts f
            LEFT JOIN temporal_facts inv ON f.invalidated_by = inv.id
            WHERE 
                (? IS NULL OR f.entity_id = ?)
                AND (? IS NULL OR f.state_key = ?)
                -- Transaction Time Filter
                AND f.recorded_at <= ?
                AND (f.invalidated_by IS NULL OR inv.recorded_at > ? OR inv.valid_from > ?)
                -- Valid Time Filter
                AND f.valid_from <= ?
                AND (f.valid_until IS NULL OR f.valid_until > ?)
        """
        params = [entity_id, entity_id, state_key, state_key, tx_time, tx_time, valid_time, valid_time, valid_time]

        if not include_tombstones:
            query += " AND f.value != 'TOMBSTONE'"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_relations(self,
                      source_id: str | None = None,
                      target_id: str | None = None,
                      relation_type: str | None = None,
                      valid_time: str | None = None,
                      tx_time: str | None = None) -> list[dict]:
        now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not valid_time:
            valid_time = now_str
        if not tx_time:
            tx_time = now_str

        query = """
            SELECT f.* FROM temporal_facts f
            LEFT JOIN temporal_facts inv ON f.invalidated_by = inv.id
            WHERE 
                f.target_entity_id IS NOT NULL
                AND (? IS NULL OR f.entity_id = ?)
                AND (? IS NULL OR f.target_entity_id = ?)
                AND (? IS NULL OR f.relation_type = ?)
                -- Transaction Time Filter
                AND f.recorded_at <= ?
                AND (f.invalidated_by IS NULL OR inv.recorded_at > ? OR inv.valid_from > ?)
                -- Valid Time Filter
                AND f.valid_from <= ?
                AND (f.valid_until IS NULL OR f.valid_until > ?)
                AND f.value != 'TOMBSTONE'
        """
        params = [source_id, source_id, target_id, target_id, relation_type, relation_type, tx_time, tx_time, valid_time, valid_time, valid_time]

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_fact_by_id(self, fact_id: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM temporal_facts WHERE id = ?;", (fact_id,)).fetchone()
            return dict(row) if row else None

    def _query_fts(self, query_str: str) -> dict[str, float]:
        cleaned = re.sub(r'[^\w\s\-\*"]', ' ', query_str)
        terms = [t for t in cleaned.split() if t]
        if not terms:
            return {}
        
        and_query = " ".join(terms)
        with self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT f.id, bm25(temporal_facts_fts) FROM temporal_facts f
                    JOIN temporal_facts_fts ON f.rowid = temporal_facts_fts.rowid
                    WHERE temporal_facts_fts MATCH ?;
                    """,
                    (and_query,)
                )
                rows = cursor.fetchall()
                if rows:
                    return {r[0]: r[1] for r in rows}
            except sqlite3.OperationalError:
                pass
                
        or_query = " OR ".join(terms)
        with self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT f.id, bm25(temporal_facts_fts) FROM temporal_facts f
                    JOIN temporal_facts_fts ON f.rowid = temporal_facts_fts.rowid
                    WHERE temporal_facts_fts MATCH ?;
                    """,
                    (or_query,)
                )
                rows = cursor.fetchall()
                return {r[0]: r[1] for r in rows}
            except sqlite3.OperationalError:
                return {}

    def _calculate_dynamic_decay(self, recorded_at_str: str, metadata: dict, now: datetime | None = None) -> float:
        if now is None:
            now = datetime.now(timezone.utc)
        
        try:
            dt_str = recorded_at_str.replace("Z", "+00:00")
            if "T" in dt_str and "+" not in dt_str and "-" not in dt_str[10:]:
                dt_str += "+00:00"
            recorded_at = datetime.fromisoformat(dt_str)
            delta = now - recorded_at
            delta_days = max(0.0, delta.total_seconds() / (24 * 3600.0))
        except Exception:
            return 1.0

        m_class = metadata.get('mutability_class')
        if m_class in (1, '1', 'permanent', 'Permanent'):
            H = float('inf')
        elif m_class in (2, '2', 'slow-drift', 'Slow-drift', 'slow_drift'):
            H = 180.0
        elif m_class in (3, '3', 'volatile', 'Volatile'):
            H = 7.0
        elif m_class in (4, '4', 'ephemeral', 'Ephemeral'):
            H = 0.0
        else:
            H = 180.0

        if H == float('inf'):
            return 1.0
        elif H == 0.0:
            return 1.0 if delta_days < (1.0 / 86400.0) else 0.0
        else:
            return 0.5 ** (delta_days / H)

    def _get_trust_weight(self, metadata: dict) -> float:
        tier = metadata.get('trust_tier') or metadata.get('provenance') or metadata.get('source')
        if not tier:
            return 0.7
        tier_lower = str(tier).lower()
        if 'user' in tier_lower or 'explicit' in tier_lower:
            return 1.0
        elif 'tool' in tier_lower or 'log' in tier_lower:
            return 0.4
        elif 'agent' in tier_lower or 'decision' in tier_lower:
            return 0.7
        return 0.7

    def _is_mfs_related(self, path_a: str | None, path_b: str | None) -> bool:
        if not path_a or not path_b:
            return True
        a = path_a.rstrip("/") + "/"
        b = path_b.rstrip("/") + "/"
        return a.startswith(b) or b.startswith(a)

    def hybrid_search(self,
                      query: str | None = None,
                      vector_results: dict[str, float] | None = None,
                      entity_id: str | None = None,
                      state_key: str | None = None,
                      mfs_path: str | None = None,
                      w_vector: float = 0.7,
                      w_lexical: float = 0.3,
                      valid_time: str | None = None,
                      tx_time: str | None = None,
                      limit: int = 5,
                      now: datetime | None = None) -> list[dict]:
        
        active_facts = self.get_facts(entity_id=entity_id, state_key=state_key, valid_time=valid_time, tx_time=tx_time)
        
        # Include facts matched by vector database that might not be in get_facts (e.g. historical/invalidated)
        if vector_results:
            active_ids = {f['id'] for f in active_facts}
            missing_ids = [fid for fid in vector_results if fid not in active_ids]
            if missing_ids:
                with self._get_connection() as conn:
                    placeholders = ",".join("?" for _ in missing_ids)
                    query_str = f"SELECT * FROM temporal_facts WHERE id IN ({placeholders});"
                    rows = conn.execute(query_str, missing_ids).fetchall()
                    active_facts.extend([dict(r) for r in rows])

        if not active_facts:
            return []
            
        fts_results = {}
        if query:
            fts_results = self._query_fts(query)
            
        candidates = []
        for f in active_facts:
            fid = f['id']
            meta = {}
            if f.get('metadata'):
                try:
                    meta = json.loads(f['metadata'])
                except Exception:
                    pass
            
            # Exclude tombstones and soft-deleted facts
            if f.get('value') == 'TOMBSTONE' or meta.get('status') == 'deleted':
                continue
                
            if query and (fid not in fts_results) and (not vector_results or fid not in vector_results):
                continue
                
            if not query and vector_results and fid not in vector_results:
                continue
                
            if mfs_path:
                cand_mfs = meta.get('mfs_path')
                if not self._is_mfs_related(mfs_path, cand_mfs):
                    continue
                    
            candidates.append((f, meta))
            
        if not candidates:
            return []
            
        scored_candidates = []
        for f, meta in candidates:
            fid = f['id']
            
            Similarity_dense = 0.0
            if vector_results and fid in vector_results:
                Similarity_dense = float(vector_results[fid])
                
            Similarity_lexical = 0.0
            if query and fid in fts_results:
                bm25_score = fts_results[fid]
                Similarity_lexical = max(0.0, -bm25_score)
                
            base_score = w_vector * Similarity_dense + w_lexical * Similarity_lexical
            decay = self._calculate_dynamic_decay(f['recorded_at'], meta, now=now)
            trust_weight = self._get_trust_weight(meta)
            final_score = base_score * decay * trust_weight
            
            scored_candidates.append({
                'fact': f,
                'metadata': meta,
                'score': final_score,
                'decay': decay,
                'trust_weight': trust_weight
            })
            
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        retrieved_ids = {c['fact']['id'] for c in scored_candidates}
        ids_to_drop = set()
        for c in scored_candidates:
            fid = c['fact']['id']
            curr_id = c['fact'].get('invalidated_by')
            while curr_id:
                if curr_id in retrieved_ids:
                    ids_to_drop.add(fid)
                    break
                if curr_id == fid:
                    break
                next_node = self.get_fact_by_id(curr_id)
                if next_node:
                    curr_id = next_node.get('invalidated_by')
                else:
                    break
                    
        pruned_candidates = [c for c in scored_candidates if c['fact']['id'] not in ids_to_drop]
        
        result_list = []
        for c in pruned_candidates[:limit]:
            item = dict(c['fact'])
            item['score'] = c['score']
            item['decay'] = c['decay']
            item['trust_weight'] = c['trust_weight']
            item['parsed_metadata'] = c['metadata']
            result_list.append(item)
            
        return result_list
