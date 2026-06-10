import unittest
import os
import sys
import sqlite3
from datetime import datetime, timezone, timedelta

# Fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../plugin")))
from plugin.sqlite_temporal import TemporalDBManager


class TestTemporalSQLite(unittest.TestCase):
    def setUp(self):
        self.db_path = "/tmp/test_temporal_mem.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.manager = TemporalDBManager(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_entity_creation(self):
        # 1. Create entities
        e_user = self.manager.add_entity("dmitry", "user")
        e_service = self.manager.add_entity("omniroute", "service")

        self.assertIsNotNone(e_user)
        self.assertIsNotNone(e_service)
        
        # 2. Get entities
        user_ent = self.manager.get_entity_by_name("dmitry")
        self.assertEqual(user_ent["id"], e_user)
        self.assertEqual(user_ent["type"], "user")

        service_ent = self.manager.get_entity_by_id(e_service)
        self.assertEqual(service_ent["name"], "omniroute")
        self.assertEqual(service_ent["type"], "service")

    def test_bi_temporal_property_updates(self):
        time_origin = datetime.now(timezone.utc)
        
        # Create entity and initial fact
        e_user = self.manager.add_entity("dmitry", "user")
        
        # T1: Dmitry sets OmniRoute port to 20130
        t1_str = (time_origin + timedelta(days=1)).isoformat(timespec="seconds")
        f1_id = self.manager.add_fact(
            entity_id=e_user,
            state_key="omniroute_port",
            value="20130",
            valid_from=t1_str,
            metadata={"source": "user"}
        )
        
        # Query facts active at T1
        facts = self.manager.get_facts(entity_id=e_user, state_key="omniroute_port", valid_time=t1_str)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["value"], "20130")
        self.assertIsNone(facts[0]["valid_until"])
        self.assertIsNone(facts[0]["invalidated_by"])

        # T2: Dmitry changes OmniRoute port to 20140
        t2_str = (time_origin + timedelta(days=5)).isoformat(timespec="seconds")
        f2_id = self.manager.add_fact(
            entity_id=e_user,
            state_key="omniroute_port",
            value="20140",
            valid_from=t2_str,
            metadata={"source": "user"}
        )

        # 1. Query for T1 (before change): Should return 20130
        facts_at_t1 = self.manager.get_facts(entity_id=e_user, state_key="omniroute_port", valid_time=t1_str)
        self.assertEqual(len(facts_at_t1), 1)
        self.assertEqual(facts_at_t1[0]["value"], "20130")

        # 2. Query for T2 (after change): Should return 20140
        facts_at_t2 = self.manager.get_facts(entity_id=e_user, state_key="omniroute_port", valid_time=t2_str)
        self.assertEqual(len(facts_at_t2), 1)
        self.assertEqual(facts_at_t2[0]["value"], "20140")
        
        # 3. Micro-audit: verify the database contains both facts (no physical deletion)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT id, value, valid_until, invalidated_by FROM temporal_facts WHERE state_key = 'omniroute_port'").fetchall()
        conn.close()
        
        self.assertEqual(len(rows), 2)
        row_map = {r[1]: r for r in rows}
        
        # First version (20130) must have valid_until=t2_str and invalidated_by=None (state transition)
        r_20130 = row_map["20130"]
        self.assertEqual(r_20130[2], t2_str)
        self.assertIsNone(r_20130[3])
        
        # Second version (20140) must have valid_until=None and invalidated_by=None
        r_20140 = row_map["20140"]
        self.assertIsNone(r_20140[2])
        self.assertIsNone(r_20140[3])

    def test_soft_delete(self):
        time_origin = datetime.now(timezone.utc)
        
        e_user = self.manager.add_entity("dmitry", "user")
        
        # T1: Add fact
        t1_str = (time_origin + timedelta(days=1)).isoformat(timespec="seconds")
        f1_id = self.manager.add_fact(
            entity_id=e_user,
            state_key="editor_preference",
            value="vim",
            valid_from=t1_str
        )
        
        # T2: Delete fact
        t2_str = (time_origin + timedelta(days=5)).isoformat(timespec="seconds")
        self.manager.delete_fact(f1_id, valid_until=t2_str)

        # 1. Query for T1: should find "vim"
        facts_at_t1 = self.manager.get_facts(entity_id=e_user, state_key="editor_preference", valid_time=t1_str)
        self.assertEqual(len(facts_at_t1), 1)
        self.assertEqual(facts_at_t1[0]["value"], "vim")

        # 2. Query for T2: should not find "vim" (now invalidated/deleted)
        facts_at_t2 = self.manager.get_facts(entity_id=e_user, state_key="editor_preference", valid_time=t2_str)
        self.assertEqual(len(facts_at_t2), 0)

        # 3. Check db records (make sure facts are still physically there - soft delete)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT id, value, valid_until, invalidated_by FROM temporal_facts WHERE state_key = 'editor_preference'").fetchall()
        conn.close()
        
        # Should have f1 (vim) and tombstone
        self.assertEqual(len(rows), 2)
        row_map = {r[1]: r for r in rows}
        
        r_vim = row_map["vim"]
        self.assertEqual(r_vim[2], t2_str)  # valid_until set to t2_str
        self.assertIsNotNone(r_vim[3])  # invalidated_by set to tombstone's ID

        r_tomb = row_map["TOMBSTONE"]
        self.assertEqual(r_tomb[2], t2_str)  # valid_until=t2_str on tombstone itself

    def test_entity_relations(self):
        time_origin = datetime.now(timezone.utc)
        
        e_user = self.manager.add_entity("dmitry", "user")
        e_service = self.manager.add_entity("plex", "service")
        
        # T1: Create relationship
        t1_str = (time_origin + timedelta(days=1)).isoformat(timespec="seconds")
        f1_id = self.manager.add_fact(
            entity_id=e_user,
            state_key="relationship",
            value="Dmitry uses Plex",
            valid_from=t1_str,
            target_entity_id=e_service,
            relation_type="uses"
        )

        # Retrieve relations
        rel_active = self.manager.get_relations(source_id=e_user, target_id=e_service, relation_type="uses", valid_time=t1_str)
        self.assertEqual(len(rel_active), 1)
        self.assertEqual(rel_active[0]["id"], f1_id)
        self.assertEqual(rel_active[0]["relation_type"], "uses")
        self.assertEqual(rel_active[0]["target_entity_id"], e_service)

if __name__ == "__main__":
    unittest.main()
