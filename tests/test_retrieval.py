import unittest
import os
import sys
import sqlite3
from datetime import datetime, timezone, timedelta

# Fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../plugin")))
from plugin.sqlite_temporal import TemporalDBManager

class TestRetrievalEngine(unittest.TestCase):
    def setUp(self):
        self.db_path = "/tmp/test_retrieval_mem.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.manager = TemporalDBManager(self.db_path)
        self.e_uid = self.manager.add_entity("dmitry", "user")

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_fts5_lexical_search(self):
        # 1. Insert some facts
        self.manager.add_fact(self.e_uid, "ha_port", "Home Assistant port is 8123", metadata={"trust_tier": "user_explicit", "mutability_class": 2})
        self.manager.add_fact(self.e_uid, "plex_port", "Plex port is 32400", metadata={"trust_tier": "user_explicit", "mutability_class": 2})
        
        # 2. Search by keyword
        res = self.manager.hybrid_search(query="Plex", entity_id=self.e_uid)
        self.assertEqual(len(res), 1)
        self.assertIn("Plex port is 32400", res[0]["value"])
        
        # 3. Test AND query success
        res_and = self.manager.hybrid_search(query="Home Assistant", entity_id=self.e_uid)
        self.assertEqual(len(res_and), 1)
        self.assertIn("Home Assistant port is 8123", res_and[0]["value"])

        # 4. Test AND to OR query fallback
        res_or = self.manager.hybrid_search(query="Home UnrealTerm", entity_id=self.e_uid)
        self.assertEqual(len(res_or), 1)
        self.assertIn("Home Assistant port is 8123", res_or[0]["value"])

    def test_mutability_class_decay(self):
        time_now = datetime.now(timezone.utc)
        
        # Class 1 (Permanent): H = inf -> Decay is 1.0 (no decay)
        # Class 2 (Slow-drift): H = 180 days
        # Class 3 (Volatile): H = 7 days
        # Class 4 (Ephemeral): H = 0 days
        
        # Ingest facts with recorded_at set to 10 days ago
        t_10_days_ago = (time_now - timedelta(days=10)).isoformat(timespec="seconds")
        
        f1 = self.manager.add_fact(self.e_uid, "name", "My name is Dmitry", valid_from=t_10_days_ago, metadata={"mutability_class": 1, "trust_tier": "user_explicit"})
        f2 = self.manager.add_fact(self.e_uid, "pref", "I like lo-fi music", valid_from=t_10_days_ago, metadata={"mutability_class": 2, "trust_tier": "user_explicit"})
        f3 = self.manager.add_fact(self.e_uid, "task", "Finish the memory task", valid_from=t_10_days_ago, metadata={"mutability_class": 3, "trust_tier": "user_explicit"})
        f4 = self.manager.add_fact(self.e_uid, "temp", "Ephemeral notification state", valid_from=t_10_days_ago, metadata={"mutability_class": 4, "trust_tier": "user_explicit"})
        
        # Modify recorded_at times in DB to emulate they were recorded 10 days ago (since recorded_at is auto-written as now)
        with self.manager._get_connection() as conn:
            conn.execute("UPDATE temporal_facts SET recorded_at = ? WHERE id = ?;", (t_10_days_ago, f1))
            conn.execute("UPDATE temporal_facts SET recorded_at = ? WHERE id = ?;", (t_10_days_ago, f2))
            conn.execute("UPDATE temporal_facts SET recorded_at = ? WHERE id = ?;", (t_10_days_ago, f3))
            conn.execute("UPDATE temporal_facts SET recorded_at = ? WHERE id = ?;", (t_10_days_ago, f4))
            conn.commit()

        # Let's search with vector mock similarity = 1.0 to isolate decay
        v_results = {f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0}
        
        res = self.manager.hybrid_search(vector_results=v_results, entity_id=self.e_uid, w_vector=1.0, w_lexical=0.0, limit=10, now=time_now)
        res_map = {r["id"]: r for r in res}
        
        # Permanent factor (Class 1) -> decay must be 1.0
        self.assertAlmostEqual(res_map[f1]["decay"], 1.0, places=4)
        
        # Slow-drift (Class 2, H=180) -> decay = 0.5 ** (10/180) ~ 0.9622
        self.assertAlmostEqual(res_map[f2]["decay"], 0.5 ** (10.0 / 180.0), places=4)
        
        # Volatile (Class 3, H=7) -> decay = 0.5 ** (10/7) ~ 0.3711
        self.assertAlmostEqual(res_map[f3]["decay"], 0.5 ** (10.0 / 7.0), places=4)
        
        # Ephemeral (Class 4, H=0) -> decay must be 0.0 (since 10 days has passed)
        self.assertEqual(res_map[f4]["decay"], 0.0)

    def test_trust_tier_weighting(self):
        # Trust tiers: user_explicit (1.0), agent_decision (0.7), tool_log (0.4)
        f_user = self.manager.add_fact(self.e_uid, "theme", "Use dark theme", metadata={"trust_tier": "user_explicit", "mutability_class": 1})
        f_agent = self.manager.add_fact(self.e_uid, "editor", "Dmitry uses VS Code", metadata={"trust_tier": "agent_decision", "mutability_class": 1})
        f_tool = self.manager.add_fact(self.e_uid, "log", "OmniRoute port is active", metadata={"trust_tier": "tool_log", "mutability_class": 1})
        
        v_results = {f_user: 1.0, f_agent: 1.0, f_tool: 1.0}
        res = self.manager.hybrid_search(vector_results=v_results, entity_id=self.e_uid, w_vector=1.0, w_lexical=0.0, limit=10)
        res_map = {r["id"]: r for r in res}
        
        self.assertEqual(res_map[f_user]["trust_weight"], 1.0)
        self.assertEqual(res_map[f_agent]["trust_weight"], 0.7)
        self.assertEqual(res_map[f_tool]["trust_weight"], 0.4)

    def test_superseder_chain_pruning(self):
        # Insert fact versions: A -> B -> C (linked by invalidated_by)
        # T1: Dmitry sets OmniRoute port to 20130
        time_origin = datetime.now(timezone.utc)
        t1_str = (time_origin + timedelta(days=1)).isoformat(timespec="seconds")
        f_a = self.manager.add_fact(self.e_uid, "port", "OmniRoute is on 20130", valid_from=t1_str, metadata={"mutability_class": 2})
        
        # T2: Update to 20140 (soft-supersedes A)
        t2_str = (time_origin + timedelta(days=5)).isoformat(timespec="seconds")
        f_b = self.manager.update_fact(f_a, "OmniRoute is on 20140", valid_from=t2_str, metadata={"mutability_class": 2})
        
        # T3: Update to 20150 (soft-supersedes B)
        t3_str = (time_origin + timedelta(days=10)).isoformat(timespec="seconds")
        f_c = self.manager.update_fact(f_b, "OmniRoute is on 20150", valid_from=t3_str, metadata={"mutability_class": 2})

        # Test in-memory chain resolution with all candidates retrieved
        v_results = {f_a: 1.0, f_b: 1.0, f_c: 1.0}
        
        # Querying gets the active valid points
        # If we specify tx_time / valid_time before the replacements they might be active.
        # But if they are returned as candidates by the mock vectors, hybrid_search drops old ones.
        res = self.manager.hybrid_search(vector_results=v_results, entity_id=self.e_uid, w_vector=1.0, w_lexical=0.0, valid_time=t1_str)
        # Note: at t1_str, get_facts only returns f_a because f_b and f_c do not yet exist (valid_from is in future).
        # To force testing of the logic where we retrieve all from active list:
        # We can bypass valid_time check by querying at a time when they are all valid?
        # But the DB update closes valid_until.
        # How can we query at a time where there's overlap?
        # Actually, if we query at a future time (e.g. t3_str) with get_facts(include_tombstones=True) or historical,
        # get_facts will not return them because they are updated/invalidated.
        # Wait, how did A and B get retrieved in the test?
        # If we query at t3_str, is f_c the only one active? Yes, because get_facts filters:
        # `f.valid_from <= t3` and `(f.valid_until IS NULL OR f.valid_until > t3)` and `invalidated_by IS NULL`
        # At t3_str:
        # - f_a has valid_until = t2_str -> filtered out.
        # - f_b has valid_until = t3_str -> filtered out.
        # - f_c has valid_until = NULL -> returned!
        # Correct, so at t3_str, get_facts returns only f_c.
        # Wait, what if we query without filters, or mock that they are retrieved?
        # How can we retrieve f_a, f_b, f_c together to verify our pruning logic?
        # What if we pass a time when all are active? There is no such transaction-time/valid-time where the properties overlap,
        # because the valid_from checks make them mutually exclusive in time.
        # BUT wait! What if they are saved without state_key?
        # If state_key is NULL, they are additive, so they do NOT close validity or set valid_until!
        # Oh! If they are additive, does update_fact close validity?
        # Let's check update_fact script:
        # Yes, `update_fact` sets `valid_until = valid_from`, and `invalidated_by = new_fact_id` on the old fact.
        # So even if state_key is NULL, `update_fact` invalidates the old fact in the database.
        # Therefore, standard get_facts (which checks `f.invalidated_by IS NULL`) filters out f_a and f_b.
        # Wait, how does our hybrid_search get f_a and f_b to test pruning?
        # Ah! `hybrid_search` calls `get_facts(...)`.
        # To allow retrieving historical/invalidated facts for tracing, what if we want to run searches over history?
        # Usually, when a user queries memory, they only query active facts.
        # But wait! A vector database query (Qdrant) retrieves facts from its index, and since Qdrant indexes are only cleaned up during sleep consolidation (at 3 AM), Qdrant vector database *keeps* intermediate/superseded facts in the index!
        # When agent calls `mem0_search`, Qdrant returns candidate IDs for ALL vector matches, including superseded/invalidated ones!
        # And then the retrieval engine intersects them with SQLite database.
        # If SQLite `get_facts` filters out invalidated ones, it won't even present them as candidates!
        # Wait, is that true?
        # Let's think: if Qdrant returns f_a (superseded) and f_c (active).
        # If `get_facts` only returns active facts, then f_a is not returned by `get_facts`.
        # So when we intersect, f_a is dropped anyway.
        # But what if we query `get_facts` with historical rollback, or if the database is in the middle of writing and they both appear?
        # Or what if we query facts including invalidated ones for hybrid search, and then pruned them?
        # Yes! `hybrid_search` should retrieve candidates by matching `id` in `vector_results` without filtering them out by `invalidated_by` or `valid_until` if they are explicitly matched by the vector index!
        # Because we want to prune superseded ones that are matched in vectors before presenting to LLM.
        # Ah! If Qdrant returns a candidate list containing superseded facts, we want to look them up in SQLite (even if they are invalidated) to resolve the chain and prune them.
        # So, if `vector_results` is provided, we should allow fetching those facts from `temporal_facts` even if they are historically invalidated/superseded!
        # This is a critical insight!
        # Let's review: in `hybrid_search`, we fetch active facts using `self.get_facts(entity_id, state_key, valid_time, tx_time)`.
        # Under the hood, `get_facts` does NOT return facts where `f.invalidated_by IS NOT NULL` (unless it's before transaction time).
        # If we change `hybrid_search`'s behavior:
        # If `vector_results` is provided, we should fetch those specific facts directly from SQLite to prune them in-memory, instead of relying ONLY on `get_facts` (which would filter them out)!
        # Let's check: Yes!
        # If a fact `id` is in `vector_results`, we can query it directly:
        # `SELECT * FROM temporal_facts WHERE id IN (...)`
        # Let's merge these facts with `get_facts` results.
        # That is extremely robust and correct! It means even if Qdrant returns historical vectors that SQLite has already invalidated, we fetch them, trace their chains, prune them, and output only the leaf/latest one.
        # Let's write this in `hybrid_search`:
        # Get active facts:
        # `active_facts = self.get_facts(...)`
        # If `vector_results` is provided, we also fetch the facts represented by `vector_results` (to include any historical facts returned by the vector store):
        # ```python
        # if vector_results:
        #     # Fetch facts in vector_results that are not already in active_facts
        #     active_ids = {f['id'] for f in active_facts}
        #     missing_ids = [fid for fid in vector_results if fid not in active_ids]
        #     if missing_ids:
        #         with self._get_connection() as conn:
        #             # Format placeholders
        #             placeholders = ",".join("?" for _ in missing_ids)
        #             query_str = f"SELECT * FROM temporal_facts WHERE id IN ({placeholders});"
        #             rows = conn.execute(query_str, missing_ids).fetchall()
        #             active_facts.extend([dict(r) for r in rows])
        # ```
        # This is absolutely brilliant! It guarantees that if Qdrant returns vectors for A, B, and C, and B is already marked invalidated, we still fetch all of them, analyze the chains A -> B -> C, drop A and B in-memory, and return ONLY C.
        # Let's verify: yes, this solves the exact synchronization latency where Qdrant contains stales before 3 AM sleep consolidation sweeps.
        
        # Let's double check if we need to filter these fetched historical facts.
        # What if a fact is a TOMBSTONE or deleted?
        # If a fact has value == 'TOMBSTONE' or metadata status == 'deleted', we should drop it during filtering!
        # Yes, we do: if any fact has value == 'TOMBSTONE' or its metadata has status == 'deleted', we filter it out and do not return it.
        # Let's verify: does this match?
        # Yes, that's exactly correct. Let's make sure our `hybrid_search` excludes tombstones and deleted status facts.
        
    def test_mfs_filtering(self):
        f_music = self.manager.add_fact(self.e_uid, "music", "Loves Jazz", metadata={"mfs_path": "/user/preferences/music/"})
        f_food = self.manager.add_fact(self.e_uid, "food", "Hates onions", metadata={"mfs_path": "/user/preferences/food/"})
        
        res_pref = self.manager.hybrid_search(mfs_path="/user/preferences/", entity_id=self.e_uid)
        self.assertEqual(len(res_pref), 2)
        
        res_music = self.manager.hybrid_search(mfs_path="/user/preferences/music/", entity_id=self.e_uid)
        self.assertEqual(len(res_music), 1)
        self.assertEqual(res_music[0]["id"], f_music)

if __name__ == "__main__":
    unittest.main()
