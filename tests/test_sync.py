import os
import sys
import unittest
import shutil
import sqlite3
import json
import uuid
from datetime import datetime, timezone

# Add script directory to sys.path to import obsidian_sqlite_sync
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from obsidian_sqlite_sync import SyncEngine, get_content_hash, parse_markdown_file, is_path_safe, utc_now, format_markdown_file

DB_PATH = "/root/mem0-temporal-hygiene/tests/test_sync.db"
VAULT_ROOT = "/root/mem0-temporal-hygiene/tests/mock_vault/"

class TestSyncEngine(unittest.TestCase):
    def setUp(self):
        # Clean up database and vault root
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        if os.path.exists(VAULT_ROOT):
            shutil.rmtree(VAULT_ROOT)
        
        self.engine = SyncEngine(db_path=DB_PATH, vault_root=VAULT_ROOT, default_entity_name="dmitry")

    def tearDown(self):
        # Clean up after tests
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        if os.path.exists(VAULT_ROOT):
            shutil.rmtree(VAULT_ROOT)

    def test_path_sanitization_and_security(self):
        """Verifies path traversal validation and namespace mapping."""
        # 1. Traversal check
        unsafe_path = os.path.join(VAULT_ROOT, "../../etc/passwd")
        self.assertFalse(is_path_safe(VAULT_ROOT, unsafe_path))
        
        safe_path = os.path.join(VAULT_ROOT, "user/preferences/music/jazz.md")
        self.assertTrue(is_path_safe(VAULT_ROOT, safe_path))
        
        # 2. Namespace mapping checks
        rel_dir = self.engine.mfs_path_to_obsidian_dir("/user/preferences/music/")
        expected_rel = os.path.join("user", "preferences", "music")
        self.assertEqual(rel_dir, expected_rel)
        
        mfs_path = self.engine.obsidian_dir_to_mfs_path(expected_rel)
        self.assertEqual(mfs_path, "/user/preferences/music/")

    def test_initial_sync_sqlite_to_obsidian(self):
        """Checks if active facts in SQLite are written down to new markdown notes in the vault."""
        # Insert a fact into the SQLite DB manually
        fact_id = str(uuid.uuid4())
        entity_id = self.engine._get_entity_id_by_name("dmitry")
        now_str = utc_now()
        
        meta = {
            "mfs_path": "/user/preferences/music/",
            "trust_tier": "user_explicit",
            "mutability_class": 2,
            "status": "active"
        }
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fact_id, entity_id, "jazz", "I like classical and jazz music.", now_str, now_str, json.dumps(meta)))
            conn.commit()
            
        # Run sync cycle
        actions = self.engine.run_sync_cycle()
        
        # Expect file created in vault
        self.assertTrue(any("Created Obsidian file" in a for a in actions))
        
        target_file = os.path.join(VAULT_ROOT, "user/preferences/music/jazz.md")
        self.assertTrue(os.path.exists(target_file))
        
        # Read file and verify frontmatter
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        metadata, body = parse_markdown_file(content)
        self.assertEqual(metadata["id"], fact_id)
        self.assertEqual(metadata["entity"], "dmitry")
        self.assertEqual(metadata["state_key"], "jazz")
        self.assertEqual(metadata["status"], "active")
        self.assertEqual(body.strip(), "I like classical and jazz music.")
        self.assertEqual(metadata["last_synced_hash"], get_content_hash(body))

    def test_import_new_obsidian_note(self):
        """Verifies that new markdown notes created in Obsidian are correctly ingested by SQLite."""
        # Create a file manually under the Vault
        rel_path = "user/preferences/food/diet.md"
        metadata = {
            "entity": "dmitry",
            "state_key": "diet",
            "trust_tier": "user_explicit",
            "mutability_class": 2,
        }
        body = "I am drinking tea now instead of coffee."
        
        self.engine.write_obsidian_file(rel_path, metadata, body)
        
        # Run sync
        actions = self.engine.run_sync_cycle()
        self.assertTrue(any("Imported new file" in a for a in actions))
        
        # Verify it exists in SQLite
        with self.engine._get_connection() as conn:
            rows = conn.execute("SELECT * FROM temporal_facts WHERE state_key = 'diet'").fetchall()
            self.assertEqual(len(rows), 1)
            row = dict(rows[0])
            self.assertEqual(row["value"], body)
            
            # YAML frontmatter of the file should now be updated with the generated UUID, timestamps, and hash
            file_path = os.path.join(VAULT_ROOT, rel_path)
            with open(file_path, "r", encoding="utf-8") as f:
                new_content = f.read()
                
            new_metadata, new_body = parse_markdown_file(new_content)
            self.assertEqual(new_metadata["id"], row["id"])
            self.assertEqual(new_metadata["status"], "active")
            self.assertEqual(new_metadata["last_synced_hash"], get_content_hash(body))

    def test_update_from_obsidian_user_edit(self):
        """Checks that editing a note in Obsidian updates the database by invalidating/superseding old fact and inserting new version."""
        # 1. Create a note initially synced
        rel_path = "user/preferences/ui/theme.md"
        fact_id = str(uuid.uuid4())
        entity_id = self.engine._get_entity_id_by_name("dmitry")
        now_str = utc_now()
        
        meta = {
            "mfs_path": "/user/preferences/ui/",
            "trust_tier": "user_explicit",
            "mutability_class": 2,
            "status": "active"
        }
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fact_id, entity_id, "theme", "I prefer light mode in summer.", now_str, now_str, json.dumps(meta)))
            conn.commit()
            
        self.engine.run_sync_cycle() # initial export
        
        # 2. Modify content in Obsidian (user edited the file)
        file_path = os.path.join(VAULT_ROOT, rel_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        metadata, body = parse_markdown_file(content)
        
        new_body = "I prefer dark mode in June."
        metadata["last_synced_hash"] = "invalid_hash_to_simulate_user_edit" # make it not match
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(format_markdown_file(metadata, new_body))
            
        # 3. Synchronize
        actions = self.engine.run_sync_cycle()
        self.assertTrue(any("Superseded SQLite Fact" in a for a in actions))
        
        # 4. Check SQLite status: old fact should be invalidated (valid_until set, status superseded)
        with self.engine._get_connection() as conn:
            # Check old fact
            old_row = conn.execute("SELECT * FROM temporal_facts WHERE id = ?", (fact_id,)).fetchone()
            self.assertIsNotNone(old_row["valid_until"])
            self.assertIsNotNone(old_row["invalidated_by"])
            old_meta = json.loads(old_row["metadata"])
            self.assertEqual(old_meta["status"], "superseded")
            
            # Check new active fact
            new_fact_id = old_row["invalidated_by"]
            new_row = conn.execute("SELECT * FROM temporal_facts WHERE id = ?", (new_fact_id,)).fetchone()
            self.assertEqual(new_row["value"], new_body)
            self.assertIsNone(new_row["valid_until"])
            new_meta = json.loads(new_row["metadata"])
            self.assertEqual(new_meta["status"], "active")
            self.assertEqual(new_meta["supersedes"], fact_id)
            
            # Check Obsidian is updated with the new ID
            with open(file_path, "r", encoding="utf-8") as f:
                final_content = f.read()
            final_meta, final_body = parse_markdown_file(final_content)
            self.assertEqual(final_meta["id"], new_fact_id)
            self.assertEqual(final_meta["last_synced_hash"], get_content_hash(new_body))

    def test_chatbot_update_in_sqlite(self):
        """Verifies that when a chatbot updates a fact in SQLite (creating a new active version), the daemon pushes this to Obsidian."""
        # 1. Add active fact and sync to Obsidian
        rel_path = "user/preferences/music/jazz.md"
        fact_id_1 = str(uuid.uuid4())
        entity_id = self.engine._get_entity_id_by_name("dmitry")
        now_str = utc_now()
        
        meta_1 = {
            "mfs_path": "/user/preferences/music/",
            "trust_tier": "user_explicit",
            "mutability_class": 2,
            "status": "active"
        }
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fact_id_1, entity_id, "jazz", "I like classical.", now_str, now_str, json.dumps(meta_1)))
            conn.commit()
            
        self.engine.run_sync_cycle()
        
        # 2. Simulate Chatbot Update in SQLite (updating/superseding fact_1 with fact_2)
        fact_id_2 = str(uuid.uuid4())
        meta_2 = {
            "mfs_path": "/user/preferences/music/",
            "trust_tier": "agent_decision",
            "mutability_class": 2,
            "status": "active",
            "supersedes": fact_id_1
        }
        
        meta_1["status"] = "superseded"
        meta_1["superseded_by"] = fact_id_2
        
        with sqlite3.connect(DB_PATH) as conn:
            # Invalidate fact_1
            conn.execute("""
                UPDATE temporal_facts 
                SET valid_until = ?, invalidated_by = ?, metadata = ? 
                WHERE id = ?
            """, (now_str, fact_id_2, json.dumps(meta_1), fact_id_1))
            
            # Insert fact_2 (new active version)
            conn.execute("""
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fact_id_2, entity_id, "jazz", "I like classical and jazz music.", now_str, now_str, json.dumps(meta_2)))
            conn.commit()
            
        # 3. Synchronize. Daemon should overwrite Obsidian note with the new text.
        actions = self.engine.run_sync_cycle()
        self.assertTrue(any("Overwrote Obsidian file" in a or "Updated Obsidian file" in a for a in actions))
        
        # 4. Check Obsidian note file metadata and body
        file_path = os.path.join(VAULT_ROOT, rel_path)
        with open(file_path, "r", encoding="utf-8") as f:
            final_content = f.read()
            
        final_meta, final_body = parse_markdown_file(final_content)
        self.assertEqual(final_meta["id"], fact_id_2)
        self.assertEqual(final_body.strip(), "I like classical and jazz music.")
        self.assertEqual(final_meta["last_synced_hash"], get_content_hash(final_body))

    def test_deletion_in_obsidian(self):
        """Verifies that removing a note in Obsidian soft-deletes the corresponding fact in SQLite."""
        # 1. Sync file to Obsidian
        rel_path = "user/preferences/music/jazz.md"
        fact_id = str(uuid.uuid4())
        entity_id = self.engine._get_entity_id_by_name("dmitry")
        
        # Use an older timestamp for recorded_at to simulate a previously synced fact
        import datetime as dt_mod
        older_dt = datetime.now(timezone.utc) - dt_mod.timedelta(hours=1)
        older_time = older_dt.isoformat(timespec="seconds")
        
        meta = {
            "mfs_path": "/user/preferences/music/",
            "trust_tier": "user_explicit",
            "mutability_class": 2,
            "status": "active"
        }
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fact_id, entity_id, "jazz", "I like jazz.", older_time, older_time, json.dumps(meta)))
            conn.commit()
            
        self.engine.run_sync_cycle()
        
        # Check target file was created
        file_path = os.path.join(VAULT_ROOT, rel_path)
        self.assertTrue(os.path.exists(file_path))
        
        # 2. Delete file in Obsidian
        os.remove(file_path)
        
        # 3. Synchronize
        actions = self.engine.run_sync_cycle()
        self.assertTrue(any("Soft-deleted SQLite Fact" in a for a in actions))
        
        # 4. Verify in SQLite that the old fact is now soft-deleted & invalid
        with self.engine._get_connection() as conn:
            old_row = conn.execute("SELECT * FROM temporal_facts WHERE id = ?", (fact_id,)).fetchone()
            self.assertIsNotNone(old_row["valid_until"])
            self.assertIsNotNone(old_row["invalidated_by"])
            old_meta = json.loads(old_row["metadata"])
            self.assertEqual(old_meta["status"], "deleted")
            
            # Check tombstone insert
            tombstone_row = conn.execute("SELECT * FROM temporal_facts WHERE id = ?", (old_row["invalidated_by"],)).fetchone()
            self.assertEqual(tombstone_row["value"], "TOMBSTONE")
            tombstone_meta = json.loads(tombstone_row["metadata"])
            self.assertEqual(tombstone_meta["status"], "deleted")
            self.assertEqual(tombstone_meta["target_fact"], fact_id)

    def test_loop_prevention(self):
        """Verifies that running sync repeatedly without modifications does not trigger any updates or loops."""
        # 1. Add active fact and sync
        fact_id = str(uuid.uuid4())
        entity_id = self.engine._get_entity_id_by_name("dmitry")
        now_str = utc_now()
        
        meta = {
            "mfs_path": "/user/preferences/music/",
            "trust_tier": "user_explicit",
            "mutability_class": 2,
            "status": "active"
        }
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fact_id, entity_id, "jazz", "I like jazz.", now_str, now_str, json.dumps(meta)))
            conn.commit()
            
        actions_1 = self.engine.run_sync_cycle()
        self.assertEqual(len(actions_1), 1) # Created Obsidian file
        
        # 2. Run sync again immediately. Expect ZERO actions (no write-loop/loops)!
        actions_2 = self.engine.run_sync_cycle()
        self.assertEqual(len(actions_2), 0, f"Sync loop detected! Actions taken: {actions_2}")
        
        # 3. Repeat again.
        actions_3 = self.engine.run_sync_cycle()
        self.assertEqual(len(actions_3), 0)

    def test_collision_resolution_obsidian_wins(self):
        """Verifies collision handling: when both Obsidian and SQLite are updated since the last sync, Obsidian wins."""
        # 1. Sync a file initially
        rel_path = "user/preferences/music/jazz.md"
        fact_id_1 = str(uuid.uuid4())
        entity_id = self.engine._get_entity_id_by_name("dmitry")
        now_str = utc_now()
        
        meta_1 = {
            "mfs_path": "/user/preferences/music/",
            "trust_tier": "user_explicit",
            "mutability_class": 2,
            "status": "active"
        }
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fact_id_1, entity_id, "jazz", "Initial jazz.", now_str, now_str, json.dumps(meta_1)))
            conn.commit()
            
        self.engine.run_sync_cycle()
        file_path = os.path.join(VAULT_ROOT, rel_path)
        
        # 2. Update Obsidian manually (simulate user editing)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        metadata_obs, body_obs = parse_markdown_file(content)
        
        user_body = "User edited jazz settings to lo-fi."
        metadata_obs["last_synced_hash"] = "force_collision_by_editing_obsidian"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(format_markdown_file(metadata_obs, user_body))
            
        # 3. Simulate Chatbot update in SQLite (remote update)
        fact_id_2 = str(uuid.uuid4())
        meta_2 = {
            "mfs_path": "/user/preferences/music/",
            "trust_tier": "agent_decision",
            "mutability_class": 2,
            "status": "active",
            "supersedes": fact_id_1
        }
        
        meta_1["status"] = "superseded"
        meta_1["superseded_by"] = fact_id_2
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE temporal_facts 
                SET valid_until = ?, invalidated_by = ?, metadata = ? 
                WHERE id = ?
            """, (now_str, fact_id_2, json.dumps(meta_1), fact_id_1))
            
            conn.execute("""
                INSERT INTO temporal_facts 
                (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fact_id_2, entity_id, "jazz", "Chatbot edited jazz settings to classical strings.", now_str, now_str, json.dumps(meta_2)))
            conn.commit()
            
        # 4. Synchronize. Collision detected. Obsidian (Master of Truth) must win!
        actions = self.engine.run_sync_cycle()
        self.assertTrue(any("Superseded SQLite Fact" in a for a in actions))
        
        # 5. Check database states: The new active fact must be based on Obsidian edits, and the chatbot version should be superseded by the user version.
        with self.engine._get_connection() as conn:
            # Get the new active fact
            active_row = conn.execute("""
                SELECT * FROM temporal_facts 
                WHERE state_key = 'jazz' 
                  AND valid_until IS NULL 
                  AND invalidated_by IS NULL
            """).fetchone()
            active_row = dict(active_row)
            
            # The active value must be the user's lo-fi choice!
            self.assertEqual(active_row["value"], user_body)
            
            # Verify the chatbot fact is invalidated by the new active row
            fact_id_3 = active_row["id"]
            chatbot_row = conn.execute("SELECT * FROM temporal_facts WHERE id = ?", (fact_id_2,)).fetchone()
            self.assertEqual(chatbot_row["invalidated_by"], fact_id_3)
            self.assertIsNotNone(chatbot_row["valid_until"])

if __name__ == "__main__":
    unittest.main()
