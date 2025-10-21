import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os


class ChatMemory:
    # Manages chat history persistence using SQLite with multi-session support
    
    def __init__(self, db_path: str = "databases/chat_history.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat_history table with session_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                user_query TEXT NOT NULL,
                model_name TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"Multi-session chat memory initialized at {self.db_path}")

    def create_new_session(self, session_name: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (session_name) VALUES (?)", (session_name,))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"Created new session '{session_name}' with ID: {session_id}")
        return session_id

    def get_all_sessions(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, session_name, created_at FROM sessions ORDER BY created_at DESC")
        sessions = [{"id": row[0], "name": row[1], "created_at": row[2]} for row in cursor.fetchall()]
        conn.close()
        return sessions

    def save_exchange(self, session_id: int, user_query: str, model_name: str, answer: str, sources: Optional[List[Dict]] = None):
        # Save a chat exchange to a specific session
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        answer_snippet = answer[:500] if len(answer) > 500 else answer
        sources_json = json.dumps(sources) if sources else None
        
        cursor.execute("""
            INSERT INTO chat_history (session_id, timestamp, user_query, model_name, answer, sources_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, timestamp, user_query, model_name, answer_snippet, sources_json))
        
        conn.commit()
        conn.close()

    def get_session_history(self, session_id: int) -> List[Dict]:
        # Get chat history for a specific session
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, user_query, model_name, answer, sources_json
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,))
        
        history = []
        for row in cursor.fetchall():
            timestamp, query, model, answer, sources_json = row
            sources = json.loads(sources_json) if sources_json else []
            
            # For simplicity, we can group user and bot messages later in the UI
            history.append({"role": "user", "content": query})
            history.append({"role": "bot", "model": model, "content": answer, "sources": sources})

        conn.close()
        return history

    def clear_history(self, session_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        print(f"Chat history for session {session_id} cleared")


    def rename_session(self, session_id: int, new_name: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE sessions SET session_name = ? WHERE id = ?", (new_name, session_id))
        conn.commit()
        conn.close()
        print(f"Renamed session {session_id} to '{new_name}'")

