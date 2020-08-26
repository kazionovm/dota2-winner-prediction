import collections
from json_reader import read_matches

# extracting features from json
def extract_features_csv(match):
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]
    
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)
        
        total_time_dead = 0
        
        if len(player['life_state']) == 3:
            total_time_dead = player['life_state']['2']
        else:
            total_time_dead = 0
        
        hero_dmg_columns = [c for c in player['damage'] if 'npc_dota_hero_' in c]
        hero_damage = sum([player['damage'][i] for i in hero_dmg_columns])
        
        hero_dmg_received_cols = [c for c in player['damage_taken'] if 'npc_dota_hero_' in c]
        hero_damage_received = sum([player['damage_taken'][i] for i in hero_dmg_received_cols])
        
        row.append((f'{player_name}_ability_level', len(player['ability_upgrades'])))
        row.append((f'{player_name}_max_hero_hit', player['max_hero_hit']['value']))
        row.append((f'{player_name}_purchase_count', len(player['purchase_log'])))
        row.append((f'{player_name}_count_ability_use', sum(player['ability_uses'].values())))
        row.append((f'{player_name}_abs_damage_dealt', sum(player['damage'].values())))
        row.append((f'{player_name}_abs_damage_received', sum(player['damage_taken'].values())))
        
        row.append((f'{player_name}_hero_damage_dealt', hero_damage))
        row.append((f'{player_name}_hero_damage_received', hero_damage_received))
        
        row.append((f'{player_name}_time_dead', total_time_dead))
        
    return collections.OrderedDict(row)

def create_features_from_jsonl(matches_file):
    df_new_features = []

    # Process raw data and add new features
    for match in read_matches(matches_file):
        features = extract_features_csv(match)

        df_new_features.append(features)

    df_new_features = pd.DataFrame.from_records(df_new_features, index='match_id_hash')
    return df_new_features

def create_new_player_features(train_data, test_data, path_to_data):
    train_new_player_features = create_features_from_jsonl(os.path.join(path_to_data, 'train_matches.jsonl')).fillna(0)
    test_new_player_features = create_features_from_jsonl(os.path.join(path_to_data, 'test_matches.jsonl')).fillna(0)

    train_features = pd.concat([train_data, train_new_player_features], axis=1)
    test_features = pd.concat([test_data, test_new_player_features], axis=1)
    
    return train_features, test_features