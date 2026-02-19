import sqlite3
import os

print("Current folder:", os.getcwd())
print("Files in folder:", os.listdir())

DB_NAME = "database.db"   # ⚠ change this to your real db name

print("Using DB:", DB_NAME)

conn = sqlite3.connect(DB_NAME)
c = conn.cursor()

c.execute("PRAGMA table_info(users)")
result = c.fetchall()

print("Users table structure:")
print(result)

conn.close()
