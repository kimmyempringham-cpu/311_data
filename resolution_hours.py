# load libraries
import sqlite3
import pandas as pd

# connect to sql data base
conn = sqlite3.connect("/Users/kimmyempringham/Downloads/311/311_database.db")
cur = conn.cursor()

# load date columns (include id for updates)
df = pd.read_sql("""
    SELECT rowid, "Date Created", "Date Last Updated"
    FROM service_requests_2019_2024
""", conn)

# convert strings to data time objects
# invalid or malformed dates will be set to NaT
df["Date Created"] = pd.to_datetime(
    df["Date Created"],
    format="%m/%d/%Y %I:%M:%S %p",
    errors="coerce"
)

df["Date Last Updated"] = pd.to_datetime(
    df["Date Last Updated"],
    format="%m/%d/%Y %I:%M:%S %p",
    errors="coerce"
)

# calculate resolution time (in hours)
df["resolution_hours"] = (
    df["Date Last Updated"] - df["Date Created"]
).dt.total_seconds() / 3600

# keep only rows with valid values
df = df.dropna(subset=["resolution_hours"])

# update sql table
updates = df[["resolution_hours", "rowid"]].values.tolist()

cur.executemany("""
    UPDATE service_requests_2019_2024
    SET resolution_hours = ?
    WHERE rowid = ?
""", updates)

conn.commit()
conn.close()
