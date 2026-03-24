"""
Schema migrations for Ragger Memory

Each migration function checks if it's needed before running.
Called from SqliteBackend._create_schema() after table creation.
"""

import json
import logging
import sqlite3

logger = logging.getLogger(__name__)


def migrate_add_dedicated_columns(conn: sqlite3.Connection, table: str = "memories"):
    """
    Add collection, category, and tags as dedicated columns.
    
    Extracts values from the JSON metadata blob into proper indexed
    columns for faster filtering. The metadata blob keeps remaining
    fields (source, filename, chunk, section, keep, bad, etc.).
    
    Schema change:
        + collection TEXT NOT NULL DEFAULT 'memory'
        + category   TEXT NOT NULL DEFAULT ''
        + tags       TEXT NOT NULL DEFAULT ''
        + INDEX idx_memories_collection
        + INDEX idx_memories_category
    """
    # Check if migration is needed
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor}
    
    if "collection" in columns:
        return  # already migrated
    
    logger.info("Migrating: adding collection, category, tags columns...")
    
    # Add columns with defaults
    conn.execute(f"ALTER TABLE {table} ADD COLUMN collection TEXT NOT NULL DEFAULT 'memory'")
    conn.execute(f"ALTER TABLE {table} ADD COLUMN category TEXT NOT NULL DEFAULT ''")
    conn.execute(f"ALTER TABLE {table} ADD COLUMN tags TEXT NOT NULL DEFAULT ''")
    
    # Create indexes
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_collection ON {table}(collection)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_category ON {table}(category)")
    
    # Backfill from JSON metadata
    cursor = conn.execute(f"SELECT id, metadata FROM {table} WHERE metadata IS NOT NULL")
    rows = cursor.fetchall()
    
    updated = 0
    for row_id, meta_json in rows:
        try:
            meta = json.loads(meta_json)
        except (json.JSONDecodeError, TypeError):
            continue
        
        collection = meta.pop("collection", "memory")
        category = meta.pop("category", "")
        
        # Tags: extract from metadata, normalize to comma-separated string
        tags_val = meta.pop("tags", "")
        if isinstance(tags_val, list):
            tag_list = [str(t) for t in tags_val]
        elif isinstance(tags_val, str) and tags_val:
            tag_list = [t.strip() for t in tags_val.split(",") if t.strip()]
        else:
            tag_list = []
        
        # Convert boolean flags to tags
        if meta.pop("keep", False):
            if "keep" not in tag_list:
                tag_list.append("keep")
        if meta.pop("bad", False):
            if "bad" not in tag_list:
                tag_list.append("bad")
        
        tags_str = ",".join(tag_list)
        
        # Write back cleaned metadata (without collection/category/tags)
        cleaned_json = json.dumps(meta) if meta else None
        
        conn.execute(
            f"UPDATE {table} SET collection = ?, category = ?, tags = ?, metadata = ? WHERE id = ?",
            (collection, category, tags_str, cleaned_json, row_id)
        )
        updated += 1
    
    conn.commit()
    logger.info(f"Migrated {updated} rows: collection/category/tags extracted from metadata")
