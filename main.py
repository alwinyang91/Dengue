
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from src.utils import data_loader, get_data_into_submission_format
from src.model_class import ModelClass
from src.drop_columns import ColumnDropperTransformer
from src.ProcessData import ProcessingData
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

# %% Load the data
df_train, df_test = data_loader()


# %% To do feature engineering
ProcessingData.duplicates_drop(df_train)
ProcessingData.fill_data(df_train, fillType ='ffill')
to_trim = ['ndvi_ne', 'ndvi_nw',
       'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm']
ProcessingData.drop_outlier(df_train,df_test, to_trim)


# %% fed the data
X = df_train.drop(columns=['total_cases'])
y = df_train['total_cases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# X_test = df_test


# %% Create pipeline
pipe = make_pipeline(
    ColumnDropperTransformer(["city", "week_start_date"]),
    StandardScaler(),
    SimpleImputer(),
    ModelClass(RandomForestRegressor())
)

# %% Fit the pipeline

pipe.fit(X_train, y_train)


print(sqrt(mean_squared_error(pipe.predict(X_test), y_test)))


# %% submit
# to do fill the data
# ProcessingData.fill_data(X_test, fillType ='ffill')
# raw_prediction = pipe.predict(X_test)
# get_data_into_submission_format(raw_prediction)