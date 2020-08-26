from json_reader import read_matches

def add_new_features(df_features, matches_file):
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']
        
        radiant_tower_kills = 0
        dire_tower_kills = 0
        radiant_barrack_kills = 0
        dire_barrack_kills = 0
        
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if objective['team'] == 2:
                    radiant_tower_kills += 1
                if objective['team'] == 3:
                    dire_tower_kills += 1
        
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_BARRACKS_KILL':
                if int(objective['key']) in {1, 2, 4, 8, 16, 32}:
                    radiant_barrack_kills += 1
                if int(objective['key']) in {64, 128, 256, 512, 1024, 2048}:
                    dire_barrack_kills += 1
                    
        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills
        
        df_features.loc[match_id_hash, 'radiant_barrack_kills'] = radiant_barrack_kills
        df_features.loc[match_id_hash, 'dire_barrack_kills'] = dire_barrack_kills
        df_features.loc[match_id_hash, 'diff_barrack_kills'] = radiant_barrack_kills - dire_barrack_kills

def create_objectives_features(train_data, test_data, path_to_data):
    add_new_features(train_data, os.path.join(path_to_data, 
                    'train_matches.jsonl'))
    
    add_new_features(test_data, 
                    os.path.join(path_to_data, 'test_matches.jsonl'))