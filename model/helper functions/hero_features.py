import pandas as pd
import numpy as np

def create_hero_dummies(train_data, test_data):
    full_df = pd.concat([train_data, test_data], sort=False)
    hero_columns = [c for c in full_df.columns if '_hero_' in c]
    train_size = train_data.shape[0]
    
    full_df = full_df[hero_columns]
    
    for team in 'r', 'd':
            players = [f'{team}{i}' for i in range(1, 6)]
            hero_cols = [f'{player}_hero_id' for player in players]
            d = pd.get_dummies(full_df[hero_cols[0]])
            
            for c in hero_cols[1:]:
                d += pd.get_dummies(full_df[c])
            
            full_df = pd.concat([full_df, d.add_prefix(f'{team}_hero_')], axis=1)
            full_df = full_df.drop(columns=hero_cols)
    
    return full_df.iloc[:train_size, :], full_df.iloc[train_size:, :]

def sum_zeros(x):
    return sum([int(elem==0) for elem in x])

def calculate_hero_wr(data, target):
    data['target'] = target.radiant_win
    rating_wins = pd.Series(0, index = range(1,121))
    rating_len = pd.Series(0, index = range(1, 121))
    
    for i in range(1, 6):
        r_hero_id_col = 'r' + str(i) + '_hero_id'
        rating_wins += data.groupby(r_hero_id_col)['target'].sum()
        rating_len += data.groupby(r_hero_id_col)['target'].count()
        
        d_hero_id_col = 'd' + str(i) + '_hero_id'
        rating_len += data.groupby(d_hero_id_col)['target'].count()
        rating_wins += data.groupby(d_hero_id_col)['target'].agg(sum_zeros)

    rating = dict.fromkeys(range(1, 121), 0)
    for key in rating.keys():
        rating[key] = rating_wins[key] / rating_len[key]
    
    return rating

def mean_team_wr(df, rating):
    df['r_win_prob'] = 0
    df['d_win_prob'] = 0

    rad_hero_cols = ['r' + str(i+1)+ '_hero_id' for i in range(5)]
    dire_hero_cols = ['d' + str(i+1)+ '_hero_id' for i in range(5)]
    df[rad_hero_cols] = df[rad_hero_cols].astype(int)
    df[dire_hero_cols] = df[dire_hero_cols].astype(int)
    for col in rad_hero_cols:
        df['r_win_prob'] += df[col].map(rating) / 5
        
    for col in dire_hero_cols:
        df['d_win_prob'] += df[col].map(rating) / 5
    
    df['win_prob'] = df['r_win_prob'] - df['d_win_prob']

def clean_data(train_data, test_data):
    hero_columns = []
    
    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        for player in players:
            hero_columns.append(f'{player}_hero_id')
    
    train_data = train_data.drop(hero_columns, axis=1)
    test_data = test_data.drop(hero_columns, axis=1)
    
    train_data = train_data.drop('target', axis=1)
    
    return train_data, test_data

def create_hero_features(train_data, test_data, train_targets):
    train_dummies, test_dummies = create_hero_dummies(train_data, test_data)
    rating = calculate_hero_wr(train_data, train_targets)
    
    mean_team_wr(train_data, rating)
    mean_team_wr(test_data, rating)
    
    train_features, test_features = clean_data(train_data, test_data)
    
    train_features = pd.concat([train_features, train_dummies], axis=1)
    test_features = pd.concat([test_features, test_dummies], axis=1)

    return train_features, test_features