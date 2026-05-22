# ════════════════════════════════════════════════════════════════
# db.py — Gestion base de données SQLite
# ════════════════════════════════════════════════════════════════
import sqlite3
import json
from datetime import datetime

DB_PATH = "historique_miction.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS bilans (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            date          TEXT NOT NULL,
            heure         TEXT NOT NULL,
            duree_miction REAL,
            type_miction  TEXT,
            n_episodes    INTEGER,
            chasse        INTEGER,
            segs_analyses INTEGER,
            labels        TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(" Base SQLite initialisée")

def sauvegarder_bilan(data):
    now = datetime.now()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO bilans
        (timestamp, date, heure, duree_miction, type_miction,
         n_episodes, chasse, segs_analyses, labels)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        now.strftime('%d/%m/%Y %H:%M:%S'),
        now.strftime('%d/%m/%Y'),
        now.strftime('%H:%M:%S'),
        data.get('duree_miction', 0),
        data.get('type_miction', '---'),
        data.get('n_episodes', 0),
        1 if data.get('chasse', False) else 0,
        data.get('segs_analyses', 0),
        json.dumps(data.get('labels', [])),
    ))
    conn.commit()
    conn.close()

def get_historique(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM bilans ORDER BY id DESC LIMIT ?', (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    for r in rows:
        r['labels'] = json.loads(r['labels'])
        r['chasse']  = bool(r['chasse'])
    return rows

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM bilans')
    total = c.fetchone()[0]
    c.execute('SELECT AVG(duree_miction) FROM bilans')
    avg_duree = round(c.fetchone()[0] or 0, 1)
    c.execute('SELECT type_miction, COUNT(*) FROM bilans GROUP BY type_miction')
    types = dict(c.fetchall())
    conn.close()
    return {'total': total, 'avg_duree': avg_duree, 'types': types}

def get_activite():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Journalier — 7 derniers jours
    c.execute('''
        SELECT date, COUNT(*) as nb,
               AVG(duree_miction) as avg_duree,
               SUM(CASE WHEN chasse=1 THEN 1 ELSE 0 END) as nb_chasse
        FROM bilans
        WHERE date >= date('now', '-6 days')
        GROUP BY date ORDER BY date ASC
    ''')
    journalier = [dict(zip(['date','nb','avg_duree','nb_chasse'], r))
                  for r in c.fetchall()]

    # Hebdomadaire — 8 dernières semaines
    c.execute('''
        SELECT strftime('%W/%Y', timestamp, 'localtime') as semaine,
               MIN(date) as debut, COUNT(*) as nb,
               AVG(duree_miction) as avg_duree
        FROM bilans
        WHERE date >= date('now', '-56 days')
        GROUP BY semaine ORDER BY semaine ASC
    ''')
    hebdo = [dict(zip(['semaine','debut','nb','avg_duree'], r))
             for r in c.fetchall()]

    # Mensuel — 6 derniers mois
    c.execute('''
        SELECT strftime('%m/%Y', timestamp, 'localtime') as mois,
               COUNT(*) as nb, AVG(duree_miction) as avg_duree,
               SUM(CASE WHEN type_miction='jet_continu' THEN 1 ELSE 0 END) as continu,
               SUM(CASE WHEN type_miction='jet_hache'   THEN 1 ELSE 0 END) as hache,
               SUM(CASE WHEN type_miction='jet_faible'  THEN 1 ELSE 0 END) as faible
        FROM bilans
        WHERE date >= date('now', '-180 days')
        GROUP BY mois ORDER BY mois ASC
    ''')
    mensuel = [dict(zip(['mois','nb','avg_duree','continu','hache','faible'], r))
               for r in c.fetchall()]

    conn.close()
    return {'journalier': journalier, 'hebdo': hebdo, 'mensuel': mensuel}

def supprimer_bilan(bilan_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM bilans WHERE id = ?', (bilan_id,))
    conn.commit()
    conn.close()