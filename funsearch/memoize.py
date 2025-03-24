import functools
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path


def _get_db_path(database_name: str):
    cache_basedir = Path.cwd() / ".memoization-cache"
    cache_basedir.mkdir(parents=True, exist_ok=True)
    return cache_basedir / f"{database_name}.db"


def _iterate_cache(database_name: str) -> Iterator[tuple]:
    """Yields (args, result) pairs from the cache."""
    with sqlite3.connect(_get_db_path(database_name)) as conn:
        cursor = conn.execute("SELECT args_str, result FROM memo_cache")
        for args_str, result_json in cursor:
            try:
                result = json.loads(result_json)  # Deserialize JSON
                yield args_str, result
            except (SyntaxError, json.JSONDecodeError):
                continue  # Skip malformed entries


def memoize(database_name: str):
    """A decorator that caches function results in a SQLite database based on input arguments."""
    db_path = _get_db_path(database_name)

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
