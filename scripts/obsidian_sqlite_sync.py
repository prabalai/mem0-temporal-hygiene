import os
import re
import uuid
import json
import hashlib
import sqlite3
import argparse
from datetime import datetime, timezone

# Helper to format ISO timestamps
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def is_path_safe(base_dir: str, target_path: str) -> bool:
    """Verifies that the target path is inside the base directory to prevent directory traversal."""
    abs_base = os.path.abspath(base_dir)
    abs_target = os.path.abspath(target_path)
    return os.path.commonpath([abs_base, abs_target]) == abs_base

def get_content_hash(text: str) -> str:
    """Computes SHA-256 hash of the normalized text body."""
    normalized = text.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def sanitize_filename(name: str) -> str:
    """Sanitizes text to be safe for filenames, keeping alphanumeric characters, hyphens, and underscores."""
    s = name.lower()
    s = re.sub(r"[^\w\-_]+", "_", s)
    s = s.strip("_")
    return s[:32]

def parse_markdown_file(content: str) -> tuple[dict, str]:
    """Splits frontmatter and body from a markdown file. Special YAML handling without external dependencies."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    
    yaml_text = parts[1]
    body = parts[2].lstrip("\n")
    
    metadata = {}
    for line in yaml_text.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            # De-quote values
            if v.startswith(('"', "'")) and v.endswith(v[0]):
                v = v[1:-1]
            if v == "null":
                v = None
            elif v == "true":
                v = True
            elif v == "false":
                v = False
            else:
                try:
                    if "." in v:
                        v = float(v)
                    else:
                        v = int(v)
                except ValueError:
                    pass
            metadata[k] = v
    return metadata, body

def format_markdown_file(metadata: dict, body: str) -> str:
    """Serializes metadata into YAML frontmatter and combines it with note body."""
    yaml_lines = ["---"]
    for k, v in metadata.items():
        if v is None:
            yaml_lines.append(f"{k}: null")
        elif isinstance(v, bool):
            yaml_lines.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, (int, float)):
            yaml_lines.append(f"{k}: {v}")
        else:
            # Escape character quotes to ensure clean YAML structure
            escaped = str(v).replace('"', '\\"')
            yaml_lines.append(f"{k}: \"{escaped}\"")
    yaml_lines.append("---")
    return "\n".join(yaml_lines) + "\n" + body

def init_db(db_path: str):
    """Initializes the SQLite database with entities and temporal_facts tables matching KUZHOMESRV specs."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                type TEXT,
                created_at TEXT NOT NULL
            );
        """)
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
        # Indexes for fast query evaluations
        conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_facts_entity ON temporal_facts(entity_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_facts_state_key ON temporal_facts(state_key);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_facts_validity ON temporal_facts(valid_from, valid_until);")
    conn.close()


class SyncEngine:
    def __init__(self, db_path: str, vault_root: str, default_entity_name: str = "dmitry"):
        self.db_path = db_path
        self.vault_root = os.path.abspath(vault_root)
        self.default_entity_name = default_entity_name
        
        # Ensure directories exist
        os.makedirs(self.vault_root, exist_ok=True)
        init_db(self.db_path)
        self._ensure_default_entity()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_default_entity(self) -> str:
        """Ensures that the default entity exists in the database."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT id FROM entities WHERE name = ?", (self.default_entity_name,)).fetchone()
            if row:
                return row["id"]
            else:
                entity_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO entities (id, name, type, created_at) VALUES (?, ?, ?, ?)",
                    (entity_id, self.default_entity_name, "user", utc_now())
                )
                conn.commit()
                return entity_id

    def _get_entity_id_by_name(self, name: str) -> str:
        """Fetch entity ID by name or create a new user entity if not exists."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
            if row:
                return row["id"]
            else:
                entity_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO entities (id, name, type, created_at) VALUES (?, ?, ?, ?)",
                    (entity_id, name, "user", utc_now())
                )
                conn.commit()
                return entity_id

    def _get_entity_name_by_id(self, entity_id: str) -> str:
        with self._get_connection() as conn:
            row = conn.execute("SELECT name FROM entities WHERE id = ?", (entity_id,)).fetchone()
            return row["name"] if row else self.default_entity_name

    def write_obsidian_file(self, rel_path: str, metadata: dict, body: str) -> str:
        """Writes note files safely, resolving paths inside the sandbox and setting WebDAV permissions (664)."""
        target_path = os.path.join(self.vault_root, rel_path)
        if not is_path_safe(self.vault_root, target_path):
            raise PermissionError(f"Secure mapping breach: path {target_path} escapes vault root!")
        
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        content = format_markdown_file(metadata, body)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Set WebDAV permissions: group-readable and writable (664)
        try:
            os.chmod(target_path, 0o664)
        except Exception:
            pass # ignore if not supported by local FS/permissions
        
        return target_path

    def mfs_path_to_obsidian_dir(self, mfs_path: str) -> str:
        """Maps an MFS namespace (e.g. /user/preferences/music/) to relative Obsidian vault path."""
        # Strip leading/trailing slashes and split
        parts = [p.strip() for p in mfs_path.split("/") if p.strip()]
        return os.path.join(*parts) if parts else ""

    def obsidian_dir_to_mfs_path(self, rel_dir: str) -> str:
        """Maps a relative folder path in Obsidian to an MFS virtual namespace path."""
        if not rel_dir or rel_dir == ".":
            return "/general/"
        normalized = rel_dir.replace("\\", "/").strip("/")
        return f"/{normalized}/"

    def determine_mutability_class(self, content: str, path: str = "") -> int:
        """Predicts mutability class: Permanent (1), Slow-drift (2), Volatile (3), Ephemeral (4)."""
        text = (content + " " + path).lower()
        if any(w in text for w in ["временн", "сейчас", "эфемир", "текущ", "ephemeral", "current", "temporary"]):
            return 4
        if any(w in text for w in ["спринт", "код ", "тест", "баг", "sprint", "task", "project", "проект", "задач"]):
            return 3
        if any(w in text for w in ["зовут", "рождени", "рождения", "birth", "name", "семья", "жена", "сын", "дочь"]):
            return 1
        return 2

    def sync_obsidian_to_sqlite(self) -> list[str]:
        """Scrapes Obsidian for updates and propagates them to SQLite."""
        actions_taken = []
        now_str = utc_now()
        
        # Check files recursively
        for root, dirs, files in os.walk(self.vault_root):
            # Ignore hidden files/folders (such as .obsidian or .git)
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            
            for file in files:
                if not file.endswith(".md") or file.startswith("."):
                    continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.vault_root)
                rel_dir = os.path.dirname(rel_path)
                mfs_path = self.obsidian_dir_to_mfs_path(rel_dir)
                
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    print(f"Error reading file {rel_path}: {e}")
                    continue
                
                metadata, body = parse_markdown_file(content)
                computed_hash = get_content_hash(body)
                
                # Check if it has an ID
                fact_id = metadata.get("id")
                
                # CASE 1: Brand new file created by user in Obsidian
                if not fact_id:
                    # New local file
                    fact_id = str(uuid.uuid4())
                    filename_without_ext = os.path.splitext(file)[0]
                    state_key = metadata.get("state_key")
                    if not state_key:
                        # Derive state_key from sanitized filename if possible, otherwise keep empty
                        if not filename_without_ext.startswith("fact_"):
                            state_key = sanitize_filename(filename_without_ext)
                        else:
                            state_key = None
                    
                    entity_name = metadata.get("entity", self.default_entity_name)
                    entity_id = self._get_entity_id_by_name(entity_name)
                    
                    mut_class = metadata.get("mutability_class") or self.determine_mutability_class(body, rel_path)
                    trust_tier = metadata.get("trust_tier", "user_explicit") # Since it was written by user in Obsidian
                    
                    meta_dict = {
                        "mfs_path": mfs_path,
                        "trust_tier": trust_tier,
                        "mutability_class": mut_class,
                        "status": "active",
                        "hash": computed_hash
                    }
                    
                    with self._get_connection() as conn:
                        conn.execute("""
                            INSERT INTO temporal_facts 
                            (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (fact_id, entity_id, state_key, body.strip(), now_str, now_str, json.dumps(meta_dict)))
                        conn.commit()
                    
                    # Write back to Obsidian with populated frontmatter
                    metadata.update({
                        "id": fact_id,
                        "entity": entity_name,
                        "state_key": state_key,
                        "trust_tier": trust_tier,
                        "mutability_class": mut_class,
                        "status": "active",
                        "valid_from": now_str,
                        "recorded_at": now_str,
                        "last_synced_hash": computed_hash
                    })
                    self.write_obsidian_file(rel_path, metadata, body)
                    actions_taken.append(f"Imported new file {rel_path} to SQLite Fact {fact_id}")
                    continue
                
                # CASE 2: Tracked file. Compare hashes to determine if user edited it
                last_synced_hash = metadata.get("last_synced_hash")
                if computed_hash != last_synced_hash:
                    # User edited the file body! Keep history and update in SQLite
                    assert fact_id is not None
                    actions_taken.extend(self._handle_obsidian_modification(str(fact_id), rel_path, metadata, body, computed_hash, now_str))
                    
        return actions_taken

    def _handle_obsidian_modification(self, fact_id: str, rel_path: str, metadata: dict, body: str, new_hash: str, now_str: str) -> list[str]:
        """Resolves modification in Obsidian by superseding the SQLite record with the new one."""
        actions = []
        with self._get_connection() as conn:
            old_row = conn.execute("SELECT * FROM temporal_facts WHERE id = ?", (fact_id,)).fetchone()
            
            # If not found in SQLite (might be database reset or deleted), we treat it as an insertion
            if not old_row:
                entity_name = metadata.get("entity", self.default_entity_name)
                entity_id = self._get_entity_id_by_name(entity_name)
                mfs_path = self.obsidian_dir_to_mfs_path(os.path.dirname(rel_path))
                
                meta_dict = {
                    "mfs_path": mfs_path,
                    "trust_tier": metadata.get("trust_tier", "user_explicit"),
                    "mutability_class": metadata.get("mutability_class", 2),
                    "status": "active",
                    "hash": new_hash
                }
                
                conn.execute("""
                    INSERT INTO temporal_facts 
                    (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (fact_id, entity_id, metadata.get("state_key"), body.strip(), now_str, now_str, json.dumps(meta_dict)))
                conn.commit()
                
                # Update file meta
                metadata["last_synced_hash"] = new_hash
                metadata["recorded_at"] = now_str
                metadata["valid_from"] = now_str
                metadata["status"] = "active"
                self.write_obsidian_file(rel_path, metadata, body)
                return [f"Re-inserted missing fact {fact_id} from Obsidian"]

            old_row = dict(old_row)
            old_meta_dict = json.loads(old_row["metadata"]) if old_row["metadata"] else {}
            
            # Check for collision.
            # We enforce "Obsidian is Master of Truth", so we link the next active item in the DB chain to be overridden as well!
            new_fact_id = str(uuid.uuid4())
            entity_id = old_row["entity_id"]
            state_key = old_row["state_key"]
            mfs_path = old_meta_dict.get("mfs_path", "/general/")
            
            # Find the active leaf/successor in the DB for this entity and key to invalidate it too
            active_leaf_row = None
            if state_key:
                active_leaf_row = conn.execute("""
                    SELECT id, metadata FROM temporal_facts
                    WHERE entity_id = ? AND state_key = ?
                      AND valid_until IS NULL AND invalidated_by IS NULL
                      AND id != ?
                """, (entity_id, state_key, fact_id)).fetchone()
            
            new_meta_dict = {
                "mfs_path": mfs_path,
                "trust_tier": metadata.get("trust_tier") or old_meta_dict.get("trust_tier", "user_explicit"),
                "mutability_class": metadata.get("mutability_class") or old_meta_dict.get("mutability_class", 2),
                "status": "active",
                "hash": new_hash,
                "supersedes": fact_id
            }
            
            # 1. Insert new version (BEGIN IMMEDIATE handles serialization via WAL configuration)
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute("""
                    INSERT INTO temporal_facts 
                    (id, entity_id, state_key, value, valid_from, recorded_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (new_fact_id, entity_id, state_key, body.strip(), now_str, now_str, json.dumps(new_meta_dict)))
                
                # 2. Invalidate old fact
                conn.execute("""
                    UPDATE temporal_facts
                    SET valid_until = ?, invalidated_by = ?
                    WHERE id = ?
                """, (now_str, new_fact_id, fact_id))
                
                # If the old fact was active, we update its metadata status to superseded
                old_meta_dict["status"] = "superseded"
                old_meta_dict["superseded_by"] = new_fact_id
                conn.execute("UPDATE temporal_facts SET metadata = ? WHERE id = ?", (json.dumps(old_meta_dict), fact_id))
                
                # 3. If there is a newer active successor, invalidate it as well (Obsidian wins over DB updates)
                if active_leaf_row:
                    leaf_id = active_leaf_row["id"]
                    conn.execute("""
                        UPDATE temporal_facts
                        SET valid_until = ?, invalidated_by = ?
                        WHERE id = ?
                    """, (now_str, new_fact_id, leaf_id))
                    
                    leaf_meta = json.loads(active_leaf_row["metadata"]) if active_leaf_row["metadata"] else {}
                    leaf_meta["status"] = "superseded"
                    leaf_meta["superseded_by"] = new_fact_id
                    conn.execute("UPDATE temporal_facts SET metadata = ? WHERE id = ?", (json.dumps(leaf_meta), leaf_id))
                
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            
            # 3. Rewrite Obsidian note with new ID, timestamps and last_synced_hash
            metadata.update({
                "id": new_fact_id,
                "status": "active",
                "valid_from": now_str,
                "recorded_at": now_str,
                "last_synced_hash": new_hash
            })
            
            # Check if we should rename the file.
            new_state_key = metadata.get("state_key")
            new_rel_path = rel_path
            if new_state_key and new_state_key != old_row["state_key"]:
                # Rename file to match the new state_key
                new_filename = f"{sanitize_filename(new_state_key)}.md"
                new_rel_path = os.path.join(os.path.dirname(rel_path), new_filename)
                
                # Delete old file
                old_file_path = os.path.join(self.vault_root, rel_path)
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)
            
            self.write_obsidian_file(new_rel_path, metadata, body)
            actions.append(f"Superseded SQLite Fact {fact_id} -> {new_fact_id} due to Obsidian edit on {new_rel_path}")
            
        return actions

    def sync_sqlite_to_obsidian(self) -> list[str]:
        """Fetches active facts from SQLite and synchronizes them to Obsidian. Marks missing ones as deleted."""
        actions_taken = []
        now_str = utc_now()
        
        # 1. Fetch active facts from SQLite
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT f.* FROM temporal_facts f
                WHERE (f.valid_until IS NULL OR f.valid_until > ?)
                  AND f.invalidated_by IS NULL
                  AND f.value != 'TOMBSTONE'
            """, (now_str,))
            active_facts = [dict(row) for row in cursor.fetchall()]

        for fact in active_facts:
            fact_id = fact["id"]
            entity_id = fact["entity_id"]
            state_key = fact["state_key"]
            value = fact["value"]
            valid_from = fact["valid_from"]
            recorded_at = fact["recorded_at"]
            
            meta_dict = json.loads(fact["metadata"]) if fact["metadata"] else {}
            status = meta_dict.get("status", "active")
            mfs_path = meta_dict.get("mfs_path", "/general/")
            trust_tier = meta_dict.get("trust_tier", "agent_decision")
            mut_class = meta_dict.get("mutability_class", 2)
            
            entity_name = self._get_entity_name_by_id(entity_id)
            
            # Map path
            obs_dir = self.mfs_path_to_obsidian_dir(mfs_path)
            if state_key:
                filename = f"{sanitize_filename(state_key)}.md"
            else:
                # Sanitized starting content + short ID
                sanitized_val = sanitize_filename(value.split("\n")[0][:20])
                if not sanitized_val:
                    filename = f"fact_{fact_id[:8]}.md"
                else:
                    filename = f"{sanitized_val}_{fact_id[:8]}.md"
                    
            rel_file_path = os.path.join(obs_dir, filename)
            abs_file_path = os.path.join(self.vault_root, rel_file_path)
            computed_db_hash = get_content_hash(value)
            
            # Check if file exists in Obsidian
            if not os.path.exists(abs_file_path):
                # Is it a deletion or a first-time write?
                rec_dt = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                age_seconds = (now_dt - rec_dt).total_seconds()
                
                if os.path.exists(os.path.join(self.vault_root, obs_dir)) and age_seconds > 5.0:
                    # Soft delete in SQLite
                    with self._get_connection() as conn:
                        tombstone_id = str(uuid.uuid4())
                        
                        conn.execute("BEGIN IMMEDIATE;")
                        try:
                            # 1. Insert tombstone
                            conn.execute("""
                                INSERT INTO temporal_facts
                                (id, entity_id, state_key, value, valid_from, valid_until, recorded_at, invalidated_by, metadata)
                                VALUES (?, ?, ?, 'TOMBSTONE', ?, ?, ?, NULL, ?)
                            """, (tombstone_id, entity_id, state_key, now_str, now_str, now_str, json.dumps({
                                "action": "delete",
                                "target_fact": fact_id,
                                "mfs_path": mfs_path,
                                "status": "deleted"
                            })))
                            
                            # 2. Invalidate the old fact
                            conn.execute("""
                                UPDATE temporal_facts
                                SET valid_until = ?, invalidated_by = ?
                                WHERE id = ?
                            """, (now_str, tombstone_id, fact_id))
                            
                            # Set status: deleted in metadata
                            meta_dict["status"] = "deleted"
                            meta_dict["deleted_by"] = tombstone_id
                            conn.execute("UPDATE temporal_facts SET metadata = ? WHERE id = ?", (json.dumps(meta_dict), fact_id))
                            conn.commit()
                        except Exception as e:
                            conn.rollback()
                            raise e
                    actions_taken.append(f"Soft-deleted SQLite Fact {fact_id} due to missing Obsidian file {rel_file_path}")
                else:
                    # Simply write the new file to Obsidian
                    metadata = {
                        "id": fact_id,
                        "entity": entity_name,
                        "state_key": state_key,
                        "trust_tier": trust_tier,
                        "mutability_class": mut_class,
                        "status": status,
                        "valid_from": valid_from,
                        "recorded_at": recorded_at,
                        "last_synced_hash": computed_db_hash
                    }
                    self.write_obsidian_file(rel_file_path, metadata, value)
                    actions_taken.append(f"Created Obsidian file {rel_file_path} for active SQLite Fact {fact_id}")
            else:
                # File exists! Parse frontmatter
                try:
                    with open(abs_file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception as e:
                    print(f"Error reading file {rel_file_path}: {e}")
                    continue
                
                metadata, body = parse_markdown_file(file_content)
                file_fact_id = metadata.get("id")
                last_synced_hash = metadata.get("last_synced_hash")
                file_body_hash = get_content_hash(body)
                
                # If they target the same fact ID
                if file_fact_id == fact_id:
                    # Compare content
                    if file_body_hash != computed_db_hash:
                        # Content differs! Did user edit Obsidian?
                        if file_body_hash == last_synced_hash:
                            # User did NOT edit Obsidian. Database is newer/updated. Write DB value to Obsidian!
                            metadata.update({
                                "last_synced_hash": computed_db_hash,
                                "recorded_at": recorded_at,
                                "valid_from": valid_from
                            })
                            self.write_obsidian_file(rel_file_path, metadata, value)
                            actions_taken.append(f"Updated Obsidian file {rel_file_path} with SQLite updates for Fact {fact_id}")
                        else:
                            # User DID edit Obsidian AND SQLite differs! Collision!
                            # Since Obsidian is the Master of Truth, we resolve by updating SQLite
                            actions_taken.extend(self._handle_obsidian_modification(str(fact_id), rel_file_path, metadata, body, file_body_hash, now_str))
                else:
                    # Different fact IDs mapping to the same file!
                    if file_body_hash == last_synced_hash:
                        # User did NOT edit the file. We can safely overwrite with SQLite's newer version.
                        new_metadata = {
                            "id": fact_id,
                            "entity": entity_name,
                            "state_key": state_key,
                            "trust_tier": trust_tier,
                            "mutability_class": mut_class,
                            "status": status,
                            "valid_from": valid_from,
                            "recorded_at": recorded_at,
                            "last_synced_hash": computed_db_hash
                        }
                        self.write_obsidian_file(rel_file_path, new_metadata, value)
                        actions_taken.append(f"Overwrote Obsidian file {rel_file_path} with newer SQLite Fact {fact_id} (replaced old Fact {file_fact_id})")
                    else:
                        # Collision: user edited the old version, but SQLite has a new version.
                        if file_fact_id:
                            actions_taken.extend(self._handle_obsidian_modification(str(file_fact_id), rel_file_path, metadata, body, file_body_hash, now_str))

        return actions_taken

    def run_sync_cycle(self) -> list[str]:
        """Runs a complete bi-directional sync cycle (Obsidian -> SQLite -> Obsidian)."""
        actions = []
        # Phase 1: Import changes from Obsidian
        actions.extend(self.sync_obsidian_to_sqlite())
        # Phase 2: Export changes from SQLite (and handle user deletions)
        actions.extend(self.sync_sqlite_to_obsidian())
        return actions


def main():
    parser = argparse.ArgumentParser(description="Obsidian ↔ SQLite Bidirectional Synchronization Daemon")
    parser.add_argument("--db-path", default=os.path.expanduser("~/.hermes/state.db"), help="Path to SQLite database")
    parser.add_argument("--vault-root", default=os.path.expanduser("~/obsidian/"), help="Path to Obsidian Vault root")
    parser.add_argument("--entity", default="dmitry", help="Default user entity name")
    parser.add_argument("--verbose", action="store_true", help="Print details of actions taken")
    args = parser.parse_args()

    print(f"🔄 Starting Sync. SQLite: {args.db_path} | Vault: {args.vault_root} | User: {args.entity}")
    engine = SyncEngine(db_path=args.db_path, vault_root=args.vault_root, default_entity_name=args.entity)
    
    try:
        actions = engine.run_sync_cycle()
        if actions:
            print(f"✅ Sync complete. Actions taken ({len(actions)}):")
            for act in actions:
                print(f"  - {act}")
        else:
            if args.verbose:
                print("♻️ Sync complete. No modifications detected.")
    except Exception as e:
        print(f"❌ Error during sync: {e}")

if __name__ == "__main__":
    main()
