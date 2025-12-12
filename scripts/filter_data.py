import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
def read_csv_file(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(file_path)

df_cities = read_csv_file('data/cities.csv')
df_countries = read_csv_file('data/countries.csv')



#count the number of each city in the cities dataframe
city_counts = df_cities['country'].value_counts()

#read daily_weather.parquet file
def read_parquet_file(file_path):
    """Reads a Parquet file and returns a pandas DataFrame."""
    return pd.read_parquet(file_path)

df_weather = read_parquet_file('data/daily_weather.parquet')


#only select weather data for stations in US
us_stations = df_cities[df_cities['country'] == 'United States of America']['station_id']
df_us_weather = df_weather[df_weather['station_id'].isin(us_stations)]
#filter the data to only include dates from 1992-01-01 to 2015-12-31
df_us_weather['date'] = pd.to_datetime(df_us_weather['date'])
df_us_weather = df_us_weather[(df_us_weather['date'] >= '1992-01-01') & (df_us_weather['date'] <= '2015-12-31')]

#write the filtered data to a new parquet file
df_us_weather.to_parquet('data/us_daily_weather_1992_2015.parquet', index=False)

#read the new parquet file to verify
df_verified = read_parquet_file('data/us_daily_weather_1992_2015.parquet')

#read sqlite database file
import sqlite3

#inspect the database and read a table
conn = sqlite3.connect('data/wildfires.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
conn.close()

def read_sqlite_db(db_path, query):
    """Reads data from a SQLite database and returns a pandas DataFrame."""
    #takes a list of names that are date columns to parse as dates
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

df_wildfires = read_sqlite_db('data/wildfires.sqlite', 'SELECT * FROM wildfires')


"""Get the following from the db (all parts of the FIRES table):
FOD_ID 
 DISCOVERY_DATE 
 STAT_CAUSE_CODE 
 STAT_CAUSE_DESCR 
 CONT_DATE 
 FIRE_SIZE 
 LATITUDE 
 LONGITUDE 
 STATE """

df_fires = read_sqlite_db(
    'data/wildfires.sqlite',
    '''
    SELECT FOD_ID, DISCOVERY_DATE, STAT_CAUSE_CODE, STAT_CAUSE_DESCR,
           CONT_DATE, FIRE_SIZE, LATITUDE, LONGITUDE, STATE
    FROM FIRES
    '''
)



# Convert Julian day numbers -> calendar dates
for col in ['DISCOVERY_DATE', 'CONT_DATE']:
    df_fires[col] = pd.to_numeric(df_fires[col], errors='coerce')
    df_fires[col] = pd.to_datetime(
        df_fires[col],
        unit='D',          # values are in days
        origin='julian',   # interpret as Julian day numbers
        errors='coerce'    # invalid / missing -> NaT
    )


#encode the date columns as datetime
df_fires['DISCOVERY_DATE'] = df_fires['DISCOVERY_DATE'].dt.date
df_fires['CONT_DATE'] = df_fires['CONT_DATE'].dt.date


#write the fires data to a new csv file
df_fires.to_csv('data/wildfires_fires_table.csv', index=False)
df_fires.to_parquet('data/wildfires_fires.parquet', index=False)

df_weather.columns

"""df_weather.columns
Index(['station_id', 'city_name', 'date', 'season', 'avg_temp_c', 'min_temp_c',
       'max_temp_c', 'precipitation_mm', 'snow_depth_mm', 'avg_wind_dir_deg',
       'avg_wind_speed_kmh', 'peak_wind_gust_kmh', 'avg_sea_level_pres_hpa',
       'sunshine_total_min'],
      dtype='object')"""


print(df_cities.columns)

