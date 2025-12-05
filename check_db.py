import sqlite3

conn = sqlite3.connect('sentiments.db')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM sentiments')
print(f'Total records: {cur.fetchone()[0]}')
cur.execute('SELECT text, sentiment, timestamp FROM sentiments LIMIT 5')
print('\nSample records:')
for r in cur.fetchall():
    print(f'  {r}')
conn.close()
