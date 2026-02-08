import os
import psycopg2

def get_connection():
    """
    Creates and returns a PostgreSQL connection using DATABASE_URL
    environment variable.
    """
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        raise Exception("DATABASE_URL environment variable is not set")

    return psycopg2.connect(db_url)


def init_db():
    """
    Initializes required tables if they do not already exist.
    Safe to call multiple times.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            time TEXT,
            value REAL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            time TEXT,
            ticker TEXT,
            action TEXT,
            shares INTEGER,
            price REAL
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
