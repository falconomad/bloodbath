
import sqlite3

conn = sqlite3.connect('/tmp/portfolio.db')
c = conn.cursor()

c.execute('CREATE TABLE IF NOT EXISTS portfolio (time TEXT, value REAL)')
c.execute('CREATE TABLE IF NOT EXISTS transactions (time TEXT, ticker TEXT, action TEXT, shares INTEGER, price REAL)')

conn.commit()
conn.close()
