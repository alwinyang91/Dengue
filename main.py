
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from src.utils import data_loader, get_data_into_submission_format
from src.model_class import ModelClass
from src.drop_columns import ColumnDropperTransformer

# %% Load the data
X_train, X_test, y_train = data_loader()



# %% Preprocessing

y = y_train.loc[:, "total_cases"]

# %% Create pipeline
pipe = make_pipeline(
    ColumnDropperTransformer(["city", "week_start_date"]),
    StandardScaler(),
    SimpleImputer(),
    ModelClass(RandomForestRegressor())
)

# %% Fit the pipeline

pipe.fit(X_train, y)

# %% Predict

raw_prediction = pipe.predict(X_test)
get_data_into_submission_format(raw_prediction)