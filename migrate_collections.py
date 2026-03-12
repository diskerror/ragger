#!/usr/bin/env python3
"""
One-time migration: tag existing memories with collection field in metadata.

Collections:
  - music: orchestration references, instrument ranges
  - sibelius: Sibelius reference, ManuScript, shortcuts, tips
  - forscore: forScore manual and related docs
  - memory: conversation memories, agent notes (default)
"""

import json
import sqlite3
import sys

DB_PATH = "/Volumes/WDBlack2/.local/share/ragger/memories.db"

# Map filename patterns to collections
COLLECTION_MAP = {
    # Music / orchestration
    "Orchestration.md": "music",
    "Principles of Orchestration": "music",
    "The Study of Orchestration": "music",
    "principlesoforch": "music",
    "Range of Instruments": "music",
    "O.md": "music",
    # Sibelius
    "Sibelius_Reference": "sibelius",
    "ManuScript_Language_Guide": "sibelius",
    "Using Sibelius": "sibelius",
    "Using_Sibelius": "sibelius",
    "Scoring Express Sibelius": "sibelius",
    "HintsTips": "sibelius",
    "Sibelius_Keyboard Shortcuts": "sibelius",
    "Basic_Shortcuts_Sibelius": "sibelius",
    "Sibelius Shortcuts": "sibelius",
    # forScore
    "forScore": "forscore",
}


def classify(metadata: dict) -> str:
    """Determine collection from metadata"""
    filename = metadata.get("filename", "")
    source = metadata.get("source", "")
    
    # Check filename first, then source
    for pattern, collection in COLLECTION_MAP.items():
        if pattern in filename or pattern in source:
            return collection
    
    # Everything else is "memory"
    return "memory"


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT id, metadata FROM memories")
    rows = cursor.fetchall()
    
    counts = {}
    updated = 0
    
    for row_id, meta_json in rows:
        meta = json.loads(meta_json) if meta_json else {}
        
        # Skip if already tagged
        if "collection" in meta:
            col = meta["collection"]
            counts[col] = counts.get(col, 0) + 1
            continue
        
        collection = classify(meta)
        meta["collection"] = collection
        counts[collection] = counts.get(collection, 0) + 1
        
        conn.execute(
            "UPDATE memories SET metadata = ? WHERE id = ?",
            (json.dumps(meta), row_id)
        )
        updated += 1
    
    conn.commit()
    conn.close()
    
    print(f"Tagged {updated} memories:")
    for col, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {col:12s}: {count:6d}")
    print(f"  {'TOTAL':12s}: {sum(counts.values()):6d}")


if __name__ == "__main__":
    main()
