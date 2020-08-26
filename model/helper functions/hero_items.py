import collections
from json_reader import read_matches

def extract_item_features_csv(match):
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]

    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)

        row.append( (f'{player_name}_items', list(map(lambda x: x['id'][5:], player['hero_inventory'])) ) )

    return collections.OrderedDict(row)

def create_item_features_from_jsonl(matches_file):
    df_new_features = []
    
    # Process raw data and add new features
    for match in read_matches(matches_file):
        features = extract_item_features_csv(match)
        df_new_features.append(features)

    df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
    return df_new_features

def add_items_dummies(train_df, test_df):
    
    full_df = pd.concat([train_df, test_df], sort=False)
    train_size = train_df.shape[0]

    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        item_columns = [f'{player}_items' for player in players]

        d = pd.get_dummies(full_df[item_columns[0]].apply(pd.Series).stack()).sum(level=0, axis=0)
        dindexes = d.index.values
        
        for c in item_columns[1:]:
            d = d.add(pd.get_dummies(full_df[c].apply(pd.Series).stack()).sum(level=0, axis=0), fill_value=0)
            d = d.loc[dindexes]

        full_df = pd.concat([full_df, d.add_prefix(f'{team}_item_')], axis=1, sort=False)
        full_df.drop(columns=item_columns, inplace=True)

    train_df = full_df.iloc[:train_size, :]
    test_df = full_df.iloc[train_size:, :]

    return train_df, test_df

def drop_consumble_items(train_df, test_df):
    
    full_df = pd.concat([train_df, test_df], sort=False)
    train_size = train_df.shape[0]

    for team in 'r', 'd':
        consumble_columns = ['tango', 'tpscroll', 
                            'bottle', 'flask',
                            'enchanted_mango', 'clarity',
                            'faerie_fire', 'ward_observer',
                            'ward_sentry']
        
        starts_with = f'{team}_item_'
        consumble_columns = [starts_with + column for column in consumble_columns]
        full_df.drop(columns=consumble_columns, inplace=True)

    train_df = full_df.iloc[:train_size, :]
    test_df = full_df.iloc[train_size:, :]

    return train_df, test_df

def create_item_features(train_data, test_data, path_to_data):
    train_items_df = create_item_features_from_jsonl(os.path.join(path_to_data, 'train_matches.jsonl')).fillna(0)
    test_items_df = create_item_features_from_jsonl(os.path.join(path_to_data, 'test_matches.jsonl')).fillna(0)
    
    train_items_df, test_items_df = add_items_dummies(train_items_df, test_items_df)
    train_items_df, test_items_df = drop_consumble_items(train_items_df, test_items_df)

    train_features = pd.concat([train_data, train_items_df], axis=1)
    test_features = pd.concat([test_data, test_items_df], axis=1)
    
    return train_features, test_features