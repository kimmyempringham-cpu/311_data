# load libraries
import sqlite3
import pandas as pd

# connect to sqlite data base
conn = sqlite3.connect("/Users/kimmyempringham/Downloads/311/311_database.db")
cur = conn.cursor()

# make sure the column exists in sqLite
# only needs to succeed once (if it already exists)
# sqlite raises an error which we safely ignore
try:
    cur.execute("""
        ALTER TABLE all_requests_demographics
        ADD COLUMN reformat_request_date TEXT
    """)
except sqlite3.OperationalError:
    # column probably already exists, ignore the error
    pass

# load existing date fields (with id so we can update the correct rows later)
df311 = pd.read_sql("""
    SELECT rowid, "Date Created", "Date Last Updated"
    FROM all_requests_demographics
""", conn)

# parse and reformat request date
# convert from original timestamp format to YYYY-MM-DD
df311["reformat_request_date"] = pd.to_datetime(
    df311["Date Created"],
    format="%m/%d/%Y %I:%M:%S %p"
).dt.strftime("%Y-%m-%d")

# keep only rows with valid values
df311 = df311.dropna(subset=["reformat_request_date"])

# prepare (date, rowid) pairs for batch update
updates = df311[["reformat_request_date", "rowid"]].values.tolist()

# update sqlite table
cur.executemany("""
    UPDATE all_requests_demographics
    SET reformat_request_date = ?
    WHERE rowid = ?
""", updates)

conn.commit()
conn.close()
