import sqlite3

try:
    conn = sqlite3.connect("cyberbully.db")
    c = conn.cursor()

    # Users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        bullying_count INTEGER DEFAULT 0,
        is_blocked INTEGER DEFAULT 0
    )
    """)

    # Comments table
    c.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        comment TEXT,
        prediction INTEGER, -- 0 = safe, 1 = bullying
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database created successfully: cyberbully.db")

except Exception as e:
    print("❌ ERROR creating DB:", e)
