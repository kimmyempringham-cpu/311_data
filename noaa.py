# load libraries
import requests
import pandas as pd

# API Set up
# insert api key here (noaa climate data)
API_KEY = "fTCbhYTzvUCCGuASUwRZGbuIsGIbyRFg"

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
HEADERS = {"token": API_KEY}

# Pull data for 2019–2024 (range end is exclusive)
years = range(2019, 2025)
all_rows = []  # list to store all daily records across years

for year in years:
    print(f"Fetching {year}...")
    offset = 1  # first record is index at one (not zero)

    while True:
        # api request parameters
        params = {
            "datasetid": "GHCND",                    # Daily weather dataset
            "stationid": "GHCND:USW00023293",        # San Jose Airport
            "startdate": f"{year}-01-01",
            "enddate": f"{year}-12-31",
            "limit": 1000,                           # max records per request
            "units": "standard",
            "datatypeid": ["PRCP", "TMAX", "TMIN", "AWND"],
            "offset": offset                         # pagination offset
        }

        # make request to noaa api
        r = requests.get(BASE_URL, params=params, headers=HEADERS)
        print(f"  Offset {offset} Status {r.status_code}")
        r.raise_for_status()

        data = r.json()
        results = data.get("results", [])

        # stop when no more records are returned
        if not results:
            print(f"  No more results for {year} at offset {offset}")
            break

        all_rows.extend(results)

        # read pagination metadata to know when to stop
        meta = data.get("metadata", {}).get("resultset", {})
        count = meta.get("count", 0)   # total available records
        limit = meta.get("limit", 1000)

        offset += limit

        if offset > count:
            break

# convert to dataframe
df = pd.DataFrame(all_rows)

# convert date column to datetime type
df["date"] = pd.to_datetime(df["date"])

# pivot dataframe so each datatype becomes its own column
# (PRCP, TMAX, TMIN, AWND)
daily = df.pivot_table(
    index="date",
    columns="datatype",
    values="value",
    aggfunc="first"
).reset_index()

daily.columns.name = None  # clean column labels

# save dataset to csv file so we can merge with 311 data
daily.to_csv(
    "/Users/kimmyempringham/Downloads/311/noaa_sanjose_2019_2024_daily.csv",
    index=False
)

print("Number of rows printed:", len(daily))
print("Date range:", daily['date'].min(), "to", daily['date'].max())
print(daily.head())
