import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

selected_features = ["['CCL', 'Floor', 'Area', 'x', 'y', 'Longitude', 'Latitude', 'wifi_hk', 'POI_density', 'Num_class', "
                     "'Num_type', 'Class_diversity', 'Type_diversity', 'MAL_Walk300', 'KDG_Walk300', 'PRS_Walk300', 'SES_Walk300', "
                     "'PAR_Walk300', 'PLG_Walk300', 'RGD_Walk300', 'MIN_Walk300', 'CPO_Walk300', 'MTA_Walk300', 'sky500', 'building500', "
                     "'vegetation500', 'NDVI500', 'NDWI500', 'NDBI500', 'AlexNet_3d1']"]
data = pd.read_csv('data.csv')

# Data split
feature_values = data[selected_features]
training_features, testing_features, training_target, testing_target = train_test_split(feature_values, data['Price'], test_size=0.2, random_state=0)

# Model fit
hyperparameters = {'bootstrap': False, 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 45}
model = ExtraTreesRegressor(**hyperparameters)
model.fit(training_features, training_target)

# Model prediction
results = model.predict(testing_features)

# Performance evaluation
print(r2_score(testing_target, results))
