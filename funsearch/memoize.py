import functools
import json
import sqlite3
from pathlib import Path


def memoize(database_name: str):
    """A decorator that caches function results in a SQLite database based on input arguments."""
    cache_basedir = Path.cwd() / ".memoization-cache"
    cache_basedir.mkdir(parents=True, exist_ok=True)
    db_path = cache_basedir / f"{database_name}.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memo_cache (
                args_str TEXT PRIMARY KEY,
                result TEXT
            )
        """)
        conn.commit()

    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def wrapper(*args):
            args_str = str(args)

            # Try to get the cached result.
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                cursor = conn.execute("SELECT result FROM memo_cache WHERE args_str = ?", (args_str,))
                row = cursor.fetchone()
                if row is not None:
                    return json.loads(row[0])

            # No cache hit; compute the result.
            result = func(*args)
            try:
                result_json = json.dumps(result)
                with sqlite3.connect(db_path, check_same_thread=False) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute(
                        "INSERT OR REPLACE INTO memo_cache(args_str, result) VALUES (?, ?)",
                        (args_str, result_json),
                    )
                    conn.commit()
            except (OSError, sqlite3.Error) as e:
                print(f"Warning: Could not write to cache database: {e}")

            return result

        return wrapper

    return decorator
