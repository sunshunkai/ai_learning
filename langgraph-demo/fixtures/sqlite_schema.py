"""创建 SQL Agent 示例使用的临时 SQLite 数据库。"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def create_database(path: str | Path) -> Path:
    database = Path(path)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                customer TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL
            );
            DELETE FROM orders;
            INSERT INTO orders (customer, amount, status) VALUES
                ('Alice', 120.5, 'paid'),
                ('Bob', 80.0, 'pending'),
                ('Alice', 45.0, 'paid');
            """
        )
    return database

