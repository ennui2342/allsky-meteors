"""Read-only access to indi-allsky's database: the sqlite file on the Pi, or
the MariaDB of a SyncAPI receiver (the cluster).

  INDI_DB=/var/lib/indi-allsky/indi-allsky.sqlite        (default; the Pi)
  INDI_DB=mysql://USER@HOST[:PORT]/DATABASE               (password in INDI_DB_PASSWORD)

Rows come back as dicts with dates as ISO strings, the form indi-allsky's
sqlite stores them in, whichever backend they came from.
"""

import os
import sqlite3
from datetime import date, datetime
from urllib.parse import unquote, urlparse

INDI_DB = os.environ.get('INDI_DB', '/var/lib/indi-allsky/indi-allsky.sqlite')


def is_mysql():
    return INDI_DB.startswith('mysql://')


def _connect():
    if is_mysql():
        import pymysql
        u = urlparse(INDI_DB)
        return pymysql.connect(host=u.hostname, port=u.port or 3306, user=unquote(u.username or ''),
                               password=os.environ.get('INDI_DB_PASSWORD', unquote(u.password or '')),
                               database=u.path.lstrip('/'), read_timeout=60,
                               cursorclass=pymysql.cursors.DictCursor)
    con = sqlite3.connect(f'file:{INDI_DB}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    return con


def _plain(v):
    if isinstance(v, datetime):
        return v.isoformat(' ')
    if isinstance(v, date):
        return v.isoformat()
    return v


def query(sql, args=()):
    """Run a read query written with sqlite's ? placeholders."""
    con = _connect()
    try:
        if is_mysql():
            with con.cursor() as cur:
                cur.execute(sql.replace('?', '%s'), args)
                rows = cur.fetchall()
        else:
            rows = con.execute(sql, args).fetchall()
        return [{k: _plain(r[k]) for k in r.keys()} for r in rows]
    finally:
        con.close()
