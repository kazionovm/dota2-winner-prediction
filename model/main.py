import numpy as np 
import pandas as pd
import os

from hero_items import create_item_features
from hero_features import create_hero_features
from json_features import create_new_player_features
from objectives_features import create_objectives_features
from feature_engineering import run_feature_engineering

from lightgbm import create_model

from submission import create_submission_file

path_to_data = "../input/mlcourse-dota2-win-prediction"

# kaggle competition data
train_features = pd.read_csv(os.path.join(path_to_data, 'train_features.csv'), index_col="match_id_hash")
test_features = pd.read_csv(os.path.join(path_to_data, 'test_features.csv'), index_col="match_id_hash")
train_targets = pd.read_csv(os.path.join(path_to_data, 'train_targets.csv'), index_col="match_id_hash")

train_features, test_features = create_item_features(train_features, test_features, path_to_data)
train_features, test_features = create_hero_features(train_features, test_features, train_targets)
train_features, test_features = create_new_player_features(train_features, test_features, path_to_data)
train_features, test_features = create_objectives_features(train_features, test_features, path_to_data)
train_features, test_features = create_new_player_features(train_features, test_features, path_to_data)
train_features, test_features = run_feature_engineering(train_features, test_features)

predictions = create_model(train_features, test_features, train_targets)

create_submission_file(test_features, predictions)