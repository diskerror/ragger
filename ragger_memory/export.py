"""
Export memories and documents from Ragger Memory

Documents: reassemble chunks back into original files with heading deduplication.
Memories: export as readable markdown, grouped by date, category, or collection.
"""

import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import SQLITE_PATH, SQLITE_MEMORIES_TABLE

logger = logging.getLogger(__name__)


def _split_heading_body(text: str) -> tuple[list[str], str]:
    """
    Split a chunk's text into heading lines and body text.
    
    During import, the full heading chain is prepended to each chunk.
    Headings are lines starting with # at the beginning of the text,
    separated from body by a blank line.
    
    Returns:
        (headings, body) where headings is a list of heading lines
        and body is the remaining text.
    """
    lines = text.split('\n')
    headings = []
    body_start = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r'^#{1,6}\s', line):
            headings.append(line)
            i += 1
            # Skip blank lines between headings
            while i < len(lines) and lines[i].strip() == '':
                i += 1
        else:
            break
    
    body = '\n'.join(lines[i:]).strip()
    return headings, body


def export_docs(collection: str, dest_dir: str, db_path: str = None):
    """
    Export a document collection back into files.
    
    Reassembles chunks by filename, deduplicates headings,
    and writes one .md file per original document.
    
    Args:
        collection: Collection name to export
        dest_dir: Destination directory
        db_path: SQLite database path (defaults to config)
    """
    path = Path(db_path or SQLITE_PATH).expanduser()
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(path))
    
    # Fetch all chunks in this collection, ordered by filename and chunk number
    cursor = conn.execute(
        f"SELECT text, metadata FROM {SQLITE_MEMORIES_TABLE} "
        f"WHERE json_extract(metadata, '$.collection') = ? "
        f"ORDER BY json_extract(metadata, '$.filename'), "
        f"json_extract(metadata, '$.chunk')",
        (collection,)
    )
    
    # Group by filename
    files: dict[str, list[tuple[str, dict]]] = {}
    for text, meta_json in cursor:
        meta = json.loads(meta_json) if meta_json else {}
        filename = meta.get('filename', meta.get('source', 'unknown.md'))
        if filename not in files:
            files[filename] = []
        files[filename].append((text, meta))
    
    conn.close()
    
    if not files:
        print(f"No documents found in collection '{collection}'")
        return
    
    print(f"Exporting {len(files)} documents from '{collection}'...")
    
    for filename, chunks in files.items():
        seen_headings: set[str] = set()
        output_parts: list[str] = []
        
        for text, meta in chunks:
            headings, body = _split_heading_body(text)
            
            # Emit only new headings from prefix
            new_headings = []
            for h in headings:
                if h not in seen_headings:
                    seen_headings.add(h)
                    new_headings.append(h)
            
            if new_headings:
                output_parts.append('\n\n'.join(new_headings))
            
            if body:
                # Strip duplicate headings from body text
                body_lines = body.split('\n')
                filtered_lines = []
                for line in body_lines:
                    stripped = line.strip()
                    if re.match(r'^#{1,6}\s', stripped) and stripped in seen_headings:
                        continue  # skip duplicate heading in body
                    if re.match(r'^#{1,6}\s', stripped):
                        seen_headings.add(stripped)
                    filtered_lines.append(line)
                body = '\n'.join(filtered_lines).strip()
                if body:
                    output_parts.append(body)
        
        # Join with double newlines
        content = '\n\n'.join(output_parts) + '\n'
        
        out_path = dest / filename
        out_path.write_text(content)
        print(f"  {filename} ({len(chunks)} chunks)")
    
    print(f"✓ Exported {len(files)} documents to {dest}")


def export_memories(dest_dir: str, group_by: str = 'date', db_path: str = None):
    """
    Export conversation memories as readable markdown.
    
    Args:
        dest_dir: Destination directory
        group_by: Grouping strategy — 'date', 'category', or 'collection'
        db_path: SQLite database path (defaults to config)
    """
    path = Path(db_path or SQLITE_PATH).expanduser()
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(path))
    
    cursor = conn.execute(
        f"SELECT text, metadata, timestamp FROM {SQLITE_MEMORIES_TABLE} "
        f"WHERE json_extract(metadata, '$.collection') = 'memory' "
        f"ORDER BY timestamp"
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No memories to export")
        return
    
    # Group entries
    groups: dict[str, list[tuple[str, dict, str]]] = {}
    
    for text, meta_json, timestamp in rows:
        meta = json.loads(meta_json) if meta_json else {}
        
        if group_by == 'date':
            # Extract date from timestamp
            try:
                dt = datetime.fromisoformat(timestamp)
                key = dt.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                key = 'unknown-date'
        elif group_by == 'category':
            key = meta.get('category', 'uncategorized')
        elif group_by == 'collection':
            key = meta.get('collection', 'memory')
        else:
            key = 'all'
        
        if key not in groups:
            groups[key] = []
        groups[key].append((text, meta, timestamp))
    
    print(f"Exporting {len(rows)} memories ({len(groups)} groups by {group_by})...")
    
    for key, entries in sorted(groups.items()):
        filename = f"{key}.md"
        parts = [f"# Memories — {key}\n"]
        
        for text, meta, timestamp in entries:
            # Build metadata header
            header_parts = []
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    header_parts.append(dt.strftime('%Y-%m-%d %H:%M UTC'))
                except (ValueError, TypeError):
                    header_parts.append(timestamp)
            
            category = meta.get('category')
            if category:
                header_parts.append(f"**{category}**")
            
            source = meta.get('source')
            if source and source not in ('chat', 'agent'):
                header_parts.append(f"Source: {source}")
            
            header = ' | '.join(header_parts)
            parts.append(f"### {header}")
            parts.append(text)
            parts.append('---')
        
        content = '\n\n'.join(parts) + '\n'
        out_path = dest / filename
        out_path.write_text(content)
        print(f"  {filename} ({len(entries)} entries)")
    
    print(f"✓ Exported {len(rows)} memories to {dest}")


def export_all(dest_dir: str, group_by: str = 'date', db_path: str = None):
    """
    Export everything: documents by collection, memories grouped.
    
    Args:
        dest_dir: Destination directory
        group_by: Grouping for memories
        db_path: SQLite database path
    """
    path = Path(db_path or SQLITE_PATH).expanduser()
    dest = Path(dest_dir)
    
    conn = sqlite3.connect(str(path))
    cursor = conn.execute(
        f"SELECT DISTINCT json_extract(metadata, '$.collection') FROM {SQLITE_MEMORIES_TABLE}"
    )
    collections = [row[0] for row in cursor if row[0]]
    conn.close()
    
    for collection in sorted(collections):
        if collection == 'memory':
            export_memories(str(dest / 'memories'), group_by, str(path))
        else:
            export_docs(collection, str(dest / collection), str(path))
