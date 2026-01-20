# import libraries for data processing & plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path

# STEP 1: READ IN DATASET FROM SQL (311 data + ACS + Weather)
# connect to sql database
#conn = sqlite3.connect("/Users/kimmyempringham/Downloads/311/311_database.db")
#conn.cursor()

# read in merged dataset
#df = pd.read_sql("""
#    SELECT *
#    FROM requests_demographics_weather
#    """, conn)

# preview dataset
#print(df.head())

# save to csv to load faster
#df.to_csv('/Users/kimmyempringham/Downloads/311/311_data_merged.csv', index = False)

# Read in csv that was saved


# STEP 2: EXPLORATORY DATA ANALYSIS
# Read in merged dataset (311 + ACS + Weather)
# use absolute path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "311_data_merged.csv"

df = pd.read_csv(DATA_PATH)

# original relative path
#df = pd.read_csv('/Users/kimmyempringham/Downloads/311/311_data_merged.csv')


# Feature engineering for EDA
df['reformat_request_date'] = pd.to_datetime(df['reformat_request_date'], errors='coerce')  # errors='coerce' avoids crashes
df = df.dropna(subset=['reformat_request_date'])  # drop rows where the date couldn't parse

df['date_only'] = df['reformat_request_date'].dt.date
df['day_of_week'] = df['reformat_request_date'].dt.day_name()
df['month_of_year'] = df['reformat_request_date'].dt.month
df['month_date'] = df['reformat_request_date'].dt.to_period('M').dt.to_timestamp()

# Focus on the top 10 most common service types
top_10_types = df['Service Type'].value_counts().head(10).index
df_top_10 = df[df['Service Type'].isin(top_10_types)].copy()  # .copy() avoids SettingWithCopy warnings


# Plot 1: Median resolution time by service type (top 10)
median_hours = df_top_10.groupby('Service Type')['resolution_hours'].median().sort_values()

plt.figure(figsize=(20, 6))
median_hours.plot(kind='barh')
plt.title('Median Hours by Request Type (Top 10)')
plt.xlabel('Median Hours')
plt.ylabel('Request Type')
plt.tight_layout()
plt.show()


# Plot 2: Percent completed within 72 hours (top 10)
perc_within_72 = df_top_10.groupby('Service Type')['under_72'].mean() * 100
perc_within_72 = perc_within_72.sort_values()

plt.figure(figsize=(20, 6))
perc_within_72.plot(kind='barh')
plt.title('Percent Requests Completed in 72 Hours (Top 10)')
plt.xlabel('Percent Completed in 72 Hours')
plt.ylabel('Request Type')
plt.tight_layout()
plt.show()


# Plot 3: Median number of requests by day of week
daily_counts = (
    df.groupby(['date_only', 'day_of_week'])['Incident_ID']
      .nunique()
      .reset_index(name='num_requests')
)

median_requests_by_day = daily_counts.groupby('day_of_week')['num_requests'].median()

weekday_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
median_requests_by_day = median_requests_by_day.reindex(weekday_order)

plt.figure(figsize=(12, 6))
median_requests_by_day.plot(kind='bar')
plt.title('Median Requests by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Requests')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Plot 4: Median percent completed within 72 hours by day of week
daily_under_72 = (
    df.groupby(['date_only', 'day_of_week'])['under_72']
      .sum()
      .reset_index(name='under_72')
)

merged_df = pd.merge(daily_under_72, daily_counts, on=['date_only', 'day_of_week'], how='inner')
merged_df['perc_under_72'] = merged_df['under_72'] / merged_df['num_requests']

under_72_by_day = merged_df.groupby('day_of_week')['perc_under_72'].median() * 100
under_72_by_day = under_72_by_day.reindex(weekday_order)

plt.figure(figsize=(12, 6))
under_72_by_day.plot(kind='bar')
plt.title('Median Percent Requests Completed in 72 Hours by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Percent Completed in 72 Hours')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Plot 5: Correlation heatmap of numeric features
num_cols = [
    'Latitude', 'Longitude', 'median_home_value', 'median_rent', 'med_hh_income',
    'per_capita_income', 'poverty_below', 'hs_grad', 'ba_degree', 'unemployed',
    'renter_occupied', 'owner_occupied', 'AWND', 'PRCP', 'TMAX', 'TMIN'
]
df_num = df[num_cols].apply(pd.to_numeric, errors='coerce')

plt.figure(figsize=(14, 12))
sns.heatmap(df_num.corr(), annot=False)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()


# Plot 6: Requests over time (monthly)
monthly_counts = (
    df.groupby('month_date')['Incident_ID']
      .nunique()
      .reset_index(name='num_requests')
)

plt.figure(figsize=(12, 5))
plt.plot(monthly_counts['month_date'], monthly_counts['num_requests'])
plt.title('Number of Requests Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Requests')
plt.tight_layout()
plt.show()

# Plot 7: Spatial differences in response time by council district
GEOJSON_PATH = BASE_DIR / "data" / "Council_District.geojson"

districts = gpd.read_file(GEOJSON_PATH).to_crs("EPSG:4326")
#districts = gpd.read_file('/Users/kimmyempringham/Downloads/Council_District.geojson').to_crs("EPSG:4326")

points = gpd.GeoDataFrame(
    df.dropna(subset=['Latitude', 'Longitude']).copy(),  # avoid invalid geometries
    geometry=gpd.points_from_xy(df.dropna(subset=['Latitude', 'Longitude'])['Longitude'],
                                df.dropna(subset=['Latitude', 'Longitude'])['Latitude']),
    crs="EPSG:4326"
)

joined = gpd.sjoin(points, districts[['DISTRICTINT', 'geometry']], how='left', predicate='within')

joined['DISTRICTINT'] = pd.to_numeric(joined['DISTRICTINT'], errors='coerce').round().astype('Int64')

median_hours_by_district = joined.groupby('DISTRICTINT')['resolution_hours'].median().sort_values()

plt.figure(figsize=(12, 6))
median_hours_by_district.plot(kind='barh')
plt.title('Median Resolution Hours by District')
plt.xlabel('Hours')
plt.ylabel('District')
plt.tight_layout()
plt.show()

# Plot 8: Average resolution hours by district for top 5 service types
# restricted to requests with valid spatial location

# drop rows without a district
joined_with_district = joined.dropna(subset=['DISTRICTINT']).copy()

# compute top 5 service types after spatial filtering
top_5_types_spatial = (joined_with_district['Service Type'].value_counts().head(5).index)

# filter data to those top 5 service types
joined_top_5 = joined_with_district[joined_with_district['Service Type'].isin(top_5_types_spatial)]

# plot
g = sns.catplot(
    data=joined_top_5,
    x="DISTRICTINT",
    y="resolution_hours",
    col="Service Type",
    kind="bar",
    height=4,
    aspect=0.8,
)

# titles / labels
g.fig.suptitle(
    'Average Resolution Hours by District\n(Top 5 Service Types with Valid Location Data)',
    y=1.05
)
g.set_axis_labels('Council District', 'Average Resolution Hours')

# iprove readability
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# Plot 9 Distribution of resolution hours by district
# cap extreme outliers for interpretability

# keep only rows with valid district and resolution time
plot_df = joined.dropna(subset=['DISTRICTINT', 'resolution_hours']).copy()

# cap resolution hours at 90 days (2160 hours)
cap_hours = 2160
plot_df = plot_df[plot_df['resolution_hours'] <= cap_hours]

# plot boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(
    x="DISTRICTINT",
    y="resolution_hours",
    data=plot_df,
    showfliers=False  # hides remaining minor outliers
)

plt.title('Distribution of Resolution Hours by District\n(Capped at 90 Days)')
plt.xlabel('Council District')
plt.ylabel('Resolution Hours')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
