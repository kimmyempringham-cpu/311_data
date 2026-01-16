# import libraries
import sqlite3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# connect to sql database
conn = sqlite3.connect("/Users/kimmyempringham/Downloads/311/311_database.db")
cur = conn.cursor()

# read in latitude and longitude (with id to update properly)
df = pd.read_sql("""
    SELECT rowid, "Latitude", "Longitude"
    FROM service_requests_2019_2024
""", conn)

# clean missing coordinates to avoid errors
df = df.dropna(subset = ["Latitude", "Longitude"])

# points_from_xy requires (x, y) which is (Longitude, Latitude)
gdf_points = gpd.GeoDataFrame(
    df,
    geometry = gpd.points_from_xy(df['Longitude'], df['Latitude']),
    crs = "EPSG:4326"
)

# load Census ZCTA file
zctas = gpd.read_file("/Users/kimmyempringham/Downloads/tl_2025_us_zcta520/tl_2025_us_zcta520.shp")

# match coordinate systems
zctas = zctas.to_crs("EPSG:4326")

# using 'within' or 'intersects' ensures the point is inside the polygon
result = gpd.sjoin(gdf_points, zctas, how="left", predicate="intersects")

# check if it looks correct
print(result[['rowid', 'Latitude', 'Longitude', 'GEOID20']].head())

zcta_column = 'GEOID20'

# drop na
df_updates = result.dropna(subset=[zcta_column])

# keep only rows with  a ZCTA
updates = df_updates[[zcta_column, "rowid"]].values.tolist()

# create zcta column in database
try:
    cur.execute('ALTER TABLE service_requests_2019_2024 ADD COLUMN zcta TEXT')
except sqlite3.OperationalError:
    # no need if column already exists
    pass

# update table to include zcta
cur.executemany("""
    UPDATE service_requests_2019_2024
    SET zcta = ?
    WHERE rowid = ?
""", updates)

conn.commit()
conn.close()
