# import libraries
import requests
import sqlite3
import pandas as pd

# insert api key (census data)
API_KEY = '03ae1dedbac9e00caf37fa6405f83e8bc35231dd'

# most recent acs 5 year = 2023
YEAR = '2023'

url = f"https://api.census.gov/data/{YEAR}/acs/acs5"

# load all acs data variables we want - start with more and filter down later
params = {
    "get": (
        "NAME,"
        "B01003_001E,"   # total population
        "B01002_001E,"   # median age
        "B11001_001E,"   # total households
        "B19013_001E,"   # median household income
        "B17001_001E,"   # poverty universe
        "B17001_002E,"   # below poverty
        "B19301_001E,"   # per capita income
        "B25064_001E,"   # median gross rent
        "B25077_001E,"   # median home value
        "B25003_001E,"   # occupied housing units
        "B25003_002E,"   # owner-occupied
        "B25003_003E,"   # renter-occupied
        "B25002_001E,"   # total housing units
        "B25002_003E,"   # vacant units
        "B23025_002E,"   # labor force
        "B23025_005E,"   # unemployed
        "B15003_001E,"   # population 25+
        "B15003_017E,"   # high school grad
        "B15003_022E,"   # bachelor's degree
        "B02001_001E,"   # total population (race table)
        "B02001_002E,"   # white alone
        "B02001_003E,"   # black alone
        "B02001_004E,"   # AIAN alone
        "B02001_005E,"   # asian alone
        "B02001_006E,"   # NHPI alone
        "B02001_007E,"   # other race alone
        "B02001_008E,"   # two or more races
        "B03003_001E,"   # total population (ethnicity)
        "B03003_003E"    # hispanic or latino
    ),
    "for": "zip code tabulation area:*",
    "key": API_KEY
}

# pull the data from census api
url = f"https://api.census.gov/data/{YEAR}/acs/acs5"
response = requests.get(url, params=params)

data = response.json()

# first row is headers
columns = data[0]
rows = data[1:]

df = pd.DataFrame(rows,columns=columns)

# need to rename for readability
rename_map = {
    "NAME": "name",
    "B01003_001E": "pop_total",
    "B01002_001E": "median_age",
    "B11001_001E": "households",
    "B19013_001E": "med_hh_income",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "poverty_below",
    "B19301_001E": "per_capita_income",
    "B25064_001E": "median_rent",
    "B25077_001E": "median_home_value",
    "B25003_001E": "occupied_units",
    "B25003_002E": "owner_occupied",
    "B25003_003E": "renter_occupied",
    "B25002_001E": "total_units",
    "B25002_003E": "vacant_units",
    "B23025_002E": "labor_force",
    "B23025_005E": "unemployed",
    "B15003_001E": "pop_25plus",
    "B15003_017E": "hs_grad",
    "B15003_022E": "ba_degree",
    "B02001_001E": "race_total",
    "B02001_002E": "white",
    "B02001_003E": "black",
    "B02001_004E": "aian",
    "B02001_005E": "asian",
    "B02001_006E": "nhpi",
    "B02001_007E": "other_race",
    "B02001_008E": "two_or_more",
    "B03003_001E": "eth_total",
    "B03003_003E": "hispanic",
    "zip code tabulation area": "zcta"
}

df = df.rename(columns=rename_map)
print(df.head())

#  save to sqlite as a table
conn = sqlite3.connect('/Users/kimmyempringham/Downloads/311/311_database.db')

df.to_sql('zcta_acs_2023', conn, if_exists='replace', index = False)
conn.close()