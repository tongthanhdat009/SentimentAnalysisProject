import sqlite3

DB_PATH = 'sentiments.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS sentiments
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()
    print('Initialized', DB_PATH)

if __name__ == '__main__':
    init_db()
