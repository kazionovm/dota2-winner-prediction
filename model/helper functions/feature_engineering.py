def ratio_transform(x):
    return max(1, x)

def death_transform(x): # used to prevent deaths column division by 0
    return max(1, x)

def gametime_transform(x): # used to prevent game_time column division by 0
    return max(1, x)

# Gold per minute
def create_GPM_feature(dataframe):
  for team in 'r', 'd':
    for i in range(1, 6):
      gpm = round((dataframe['%s%d_gold' % (team, i)] / 
                 dataframe['game_time'].apply(gametime_transform)) * 60, 2)
        
        # evaluating gpm for every player
      dataframe['%s%d_GPM' % (team, i)] = gpm
    
    # Creating team average GPM feature
    avg_gpm = round((dataframe['%s1_GPM' % team] + dataframe['%s2_GPM' % team] +
                    dataframe['%s3_GPM' % team] + dataframe['%s4_GPM' % team] +
                    dataframe['%s5_GPM' % team]) / 5, 2)
    
    dataframe['%s_avgGPM' % team] = avg_gpm
    
    players = [f'{team}{i}_GPM' for i in range(1, 6)]
    dataframe['%s_maxGPM' % team] = dataframe[players].max(axis=1)
    dataframe['%s_minGPM' % team] = dataframe[players].min(axis=1)
    dataframe['%s_stdGPM' % team] = dataframe[players].std(axis=1)
  
  dataframe['GPM_ratio'] = dataframe['r_avgGPM'] / (dataframe['d_avgGPM'].apply(ratio_transform))

# KDA
def create_KDA_feature(dataframe):
  for team in 'r', 'd':
    for i in range(1, 6):
      kda = round((dataframe['%s%d_kills' % (team, i)] + 
                  dataframe['%s%d_assists' % (team, i)]) / (dataframe['%s%d_deaths' % (team, i)].apply(death_transform)), 2)
    
      # evaluating kda for every player
      dataframe['%s%d_KDA' % (team, i)] = kda

    # Creating team average KDA feature
    avg_kda = round((dataframe['%s1_KDA' % team] + dataframe['%s2_KDA' % team] + 
                    dataframe['%s3_KDA' % team] + dataframe['%s4_KDA' % team] + 
                    dataframe['%s5_KDA' % team]) / 5, 2)

    
    dataframe['%s_avgKDA' % team] = avg_kda
    
    players = [f'{team}{i}_KDA' for i in range(1, 6)]
    dataframe['%s_maxKDA' % team] = dataframe[players].max(axis=1)
    dataframe['%s_minKDA' % team] = dataframe[players].min(axis=1)
    dataframe['%s_stdKDA' % team] = dataframe[players].std(axis=1)
  
  dataframe['KDA_ratio'] = dataframe['r_avgKDA'] / (dataframe['d_avgKDA'].apply(ratio_transform))

# Level
def create_lvl_features(dataframe):
  for team in 'r', 'd':
    avg_level = round((dataframe['%s1_level' % team] + dataframe['%s2_level' % team] + 
                 dataframe['%s3_level' % team] + dataframe['%s4_level' % team] +
                 dataframe['%s5_level' % team]) / 5)
    
    total_level = (dataframe['%s1_level' % team] + dataframe['%s2_level' % team] +
                   dataframe['%s3_level' % team] + dataframe['%s4_level' % team] +
                   dataframe['%s5_level' % team])
    
    dataframe['%s_avg_level' % team] = avg_level
    dataframe['%s_total_level' % team] = total_level
    
    players = [f'{team}{i}_level' for i in range(1, 6)]
    dataframe['%s_max_level' % team] = dataframe[players].max(axis=1)
    dataframe['%s_min_level' % team] = dataframe[players].min(axis=1)
  
  dataframe['level_ratio'] = dataframe['r_avg_level'] / dataframe['d_avg_level']

# XPM
def create_XPM_feature(dataframe):
  for team in 'r', 'd':
    for i in range(1, 6):
      xpm = round((dataframe['%s%d_xp' % (team, i)] / 
                 dataframe['game_time'].apply(gametime_transform)) * 60, 2)
        
      # evaluating xpm for every player
      dataframe['%s%d_XPM' % (team, i)] = xpm
      
    # Creating team average XPM feature
    avg_xpm = round((dataframe['%s1_XPM' % team] + dataframe['%s2_XPM' % team] +
                    dataframe['%s3_XPM' % team] + dataframe['%s4_XPM' % team] +
                    dataframe['%s5_XPM' % team]) / 5, 2)
    
    dataframe['%s_avgXPM' % team] = avg_xpm
    
    players = [f'{team}{i}_XPM' for i in range(1, 6)]
    dataframe['%s_maxXPM' % team] = dataframe[players].max(axis=1)
    dataframe['%s_minXPM' % team] = dataframe[players].min(axis=1)
    dataframe['%s_stdXPM' % team] = dataframe[players].std(axis=1)
  
  dataframe['XPM_ratio'] = dataframe['r_avgXPM'] / (dataframe['d_avgXPM'].apply(ratio_transform))

# KPM
def create_KPM_feature(dataframe):
  for team in 'r', 'd':
    for i in range(1, 6):
        kpm = round((dataframe['%s%d_kills' % (team, i)] / 
                 dataframe['game_time'].apply(gametime_transform)) * 60, 2)
        
        # evaluating xpm for every player
        dataframe['%s%d_KPM' % (team, i)] = kpm
    
    # Creating team average KPM feature
    avg_kpm = round((dataframe['%s1_KPM' % team] + dataframe['%s2_KPM' % team] +
                    dataframe['%s3_KPM' % team] + dataframe['%s4_KPM' % team] +
                    dataframe['%s5_KPM' % team]) / 5, 2)
    
    dataframe['%s_avgKPM' % team] = avg_kpm
    
    players = [f'{team}{i}_KPM' for i in range(1, 6)]
    dataframe['%s_maxKPM' % team] = dataframe[players].max(axis=1)
    dataframe['%s_minKPM' % team] = dataframe[players].min(axis=1)
    dataframe['%s_stdKPM' % team] = dataframe[players].std(axis=1)
  
  dataframe['KPM_ratio'] = dataframe['r_avgKPM'] / (dataframe['d_avgKPM'].apply(ratio_transform))

# Networth
def create_networth_features(dataframe):
  for team in 'r', 'd':
    total_gold = (dataframe['%s1_gold' % team] + dataframe['%s2_gold' % team] +
                  dataframe['%s3_gold' % team] + dataframe['%s4_gold' % team] +
                  dataframe['%s5_gold' % team])
    
    total_xp = (dataframe['%s1_xp' % team] + dataframe['%s2_xp' % team] +
                dataframe['%s3_xp' % team] + dataframe['%s4_xp' % team] +
                dataframe['%s5_xp' % team])
    
    dataframe['%s_total_gold' % team] = total_gold
    dataframe['%s_total_xp' % team] = total_xp
    
    players = [f'{team}{i}_gold' for i in range(1, 6)]
    dataframe['%s_max_gold' % team] = dataframe[players].max(axis=1)
    dataframe['%s_min_gold' % team] = dataframe[players].min(axis=1)
    dataframe['%s_std_gold' % team] = dataframe[players].std(axis=1)
  
  dataframe['total_gold_ratio'] = dataframe['r_total_gold'] / (dataframe['d_total_gold'].apply(ratio_transform))
  dataframe['total_xp_ratio'] = dataframe['r_total_xp'] / (dataframe['d_total_xp'].apply(ratio_transform))

# Total Rune picked & Total roshans killed
def create_totalr_features(dataframe):
  for team in 'r', 'd':
    total_rune_pickups = 0
    total_roshans_killed = 0
    
    for i in range(1, 6):
        total_rune_pickups += dataframe['%s%d_rune_pickups' % (team, i)]
        total_roshans_killed += dataframe['%s%d_roshans_killed' % (team, i)]
    
    dataframe['%s_total_rune_pickups' % team] = total_rune_pickups
    dataframe['%s_total_roshans_killed' % team] = total_roshans_killed
  
  dataframe['diff_roshans_kills'] = dataframe['r_total_roshans_killed'] - dataframe['d_total_roshans_killed']
  dataframe['roshans_kills_ratio'] = (dataframe['r_total_roshans_killed'] /
                                      (dataframe['d_total_roshans_killed'].apply(ratio_transform)))

# Health
def create_health_features(dataframe):
  for team in 'r', 'd':
    team_total_health = 0
    
    for i in range(1, 6):
        team_total_health += dataframe['%s%d_max_health' % (team, i)]
    
    team_avg_health = round(team_total_health / 5, 2)
    
    dataframe['%s_total_health' % team] = team_total_health
    dataframe['%s_avg_health' % team] = team_avg_health
    
    players = [f'{team}{i}_health' for i in range(1, 6)]
    dataframe['%s_max_health' % team] = dataframe[players].max(axis=1)
    dataframe['%s_min_health' % team] = dataframe[players].min(axis=1)
    dataframe['%s_std_health' % team] = dataframe[players].std(axis=1)
  
  dataframe['total_health_ratio'] = dataframe['r_total_health'] / (dataframe['d_total_health'].apply(ratio_transform))
  dataframe['avg_health_ratio'] = dataframe['r_avg_health'] / (dataframe['d_avg_health'].apply(ratio_transform))

# Wards
def create_wards_features(dataframe):
  for team in 'r', 'd':
    total_obs_placed = 0
    total_sen_placed = 0
    
    for i in range(1, 6):
        total_obs_placed += dataframe['%s%d_obs_placed' % (team, i)]
        total_sen_placed += dataframe['%s%d_sen_placed' % (team, i)]
    
    avg_obs_placed = total_obs_placed / 5
    avg_sen_placed = total_sen_placed / 5
    
    total_wards_placed = total_obs_placed + total_sen_placed
    total_avg_wards = avg_obs_placed + avg_sen_placed
    
    dataframe['%s_total_obs_placed' % team] = total_obs_placed
    dataframe['%s_total_sen_placed' % team] = total_sen_placed
    
    dataframe['%s_avg_obs_placed' % team] = avg_obs_placed
    dataframe['%s_avg_sen_placed' % team] = avg_sen_placed
    
    dataframe['%s_total_wards_placed' % team] = total_wards_placed
    dataframe['%s_total_avg_wards_placed' % team] = total_avg_wards

# Teamfights
def create_teamfight_features(dataframe):
  for team in 'r', 'd':
    total_teamfight_participation = 0
    
    for i in range(1, 5):
        total_teamfight_participation += dataframe['%s%d_teamfight_participation' % (team, i)]
    
    avg_teamfight_participation = round((total_teamfight_participation / 5), 2)
    dataframe['%s_total_teamfight_participation' % team] = total_teamfight_participation
    dataframe['%s_avg_teamfight_participation' % team] = avg_teamfight_participation
    
    players = [f'{team}{i}_teamfight_participation' for i in range(1, 6)]
    dataframe['%s_max_teamfight_participation' % team] = dataframe[players].max(axis=1)
    dataframe['%s_min_teamfight_participation' % team] = dataframe[players].min(axis=1)
    dataframe['%s_std_teamfight_participation' % team] = dataframe[players].std(axis=1)
  
  dataframe['total_teamfight_ratio'] = (dataframe['r_total_teamfight_participation'] / 
                                          (dataframe['d_total_teamfight_participation'].apply(ratio_transform)))

# Damage
def create_HDM_feature(dataframe):
  for team in 'r', 'd':
    total_damage = 0
    for i in range(1, 6):
        total_damage += dataframe['%s%d_hero_damage_dealt' % (team, i)]
        
        hdm = round((dataframe['%s%d_hero_damage_dealt' % (team, i)] / 
                 dataframe['game_time'].apply(gametime_transform)) * 60, 2)
        
        # evaluating xpm for every player
        dataframe['%s%d_HDM' % (team, i)] = hdm
    
    total_hdm = (dataframe['%s1_HDM' % team] + dataframe['%s2_HDM' % team] +
                dataframe['%s3_HDM' % team] + dataframe['%s4_HDM' % team] +
                dataframe['%s5_HDM' % team])
    # Creating team average HDM feature
    avg_hdm = round((total_hdm/5), 2)
    
    dataframe['%s_totalHDM' % team] = total_hdm
    dataframe['%s_avgHDM' % team] = avg_hdm
    
    dataframe['%s_totalHeroDamage' % team] = total_damage
    
    players = [f'{team}{i}_hero_damage_dealt' for i in range(1, 6)]
    dataframe['%s_max_hero_damage_dealt' % team] = dataframe[players].max(axis=1)
    dataframe['%s_min_hero_damage_dealt' % team] = dataframe[players].min(axis=1)
    dataframe['%s_std_hero_damage_dealt' % team] = dataframe[players].std(axis=1)
  
  dataframe['total_heroDamage_ratio'] = (dataframe['r_totalHeroDamage'] / 
                                        (dataframe['d_totalHeroDamage'].apply(ratio_transform)))

# Time dead
def create_td_feature(dataframe):
  for team in 'r', 'd':
    total_time_dead = 0
    for i in range(1, 6):
        total_time_dead += round((dataframe['%s%d_time_dead' % (team, i)] / 60), 2)
        dataframe['%s%d_time_dead' % (team, i)] = round((dataframe['%s%d_time_dead' % (team, i)] / 60), 2)
    
    avg_time_dead = round((total_time_dead/5), 2)
    
    dataframe['%s_total_time_dead' % team] = total_time_dead
    dataframe['%s_avg_time_dead' % team] = avg_time_dead
  
  dataframe['total_td_ratio'] = (dataframe['r_total_time_dead'] / 
                                (dataframe['d_total_time_dead'].apply(ratio_transform)))
  dataframe['avg_td_ratio'] = (dataframe['r_avg_time_dead'] / 
                              (dataframe['d_avg_time_dead'].apply(ratio_transform)))

# Hero hit
def create_hero_hit_features(dataframe):
  for team in 'r', 'd':
    total_hero_hit = 0
    
    for i in range(1, 5):
        total_hero_hit += dataframe['%s%d_max_hero_hit' % (team, i)]
    
    avg_hero_hit = round((total_hero_hit / 5), 2)
    dataframe['%s_total_hero_hit' % team] = total_hero_hit
    dataframe['%s_avg_hero_hit' % team] = avg_hero_hit
    
    players = [f'{team}{i}_max_hero_hit' for i in range(1, 6)]
    dataframe['%s_max_hit' % team] = dataframe[players].max(axis=1)
    dataframe['%s_min_hit' % team] = dataframe[players].min(axis=1)
    dataframe['%s_std_hit' % team] = dataframe[players].std(axis=1)
  
  dataframe['hit_ratio'] = dataframe['r_avg_hero_hit']/dataframe['d_avg_hero_hit']

# Coordinates
def make_coordinate_features(dataframe):
    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        for player in players:
            dataframe[f'{player}_distance'] = np.sqrt(dataframe[f'{player}_x']**2 + dataframe[f'{player}_y']**2)
            dataframe.drop(columns=[f'{player}_x', f'{player}_y'], inplace=True)

        dataframe['%s_max_distance' % team] = dataframe[players].max(axis=1)
        dataframe['%s_min_distance' % team] = dataframe[players].min(axis=1)
        dataframe['%s_std_distance' % team] = dataframe[players].std(axis=1)
        
    return dataframe

def run_feature_engineering(train_data, test_data):
  create_GPM_feature(train_data)
  create_GPM_feature(test_data)
  
  create_KDA_feature(train_data)
  create_KDA_feature(test_data)
  
  create_lvl_features(train_data)
  create_lvl_features(test_data)
  
  create_XPM_feature(train_data)
  create_XPM_feature(test_data)
  
  create_KPM_feature(train_data)
  create_KPM_feature(test_data)
  
  create_networth_features(train_data)
  create_networth_features(test_data)
  
  create_totalr_features(train_data)
  create_totalr_features(test_data)
  
  create_health_features(train_data)
  create_health_features(test_data)
  
  create_wards_features(train_data)
  create_wards_features(test_data)
  
  create_teamfight_features(train_data)
  create_teamfight_features(test_data)
  
  create_HDM_feature(train_data)
  create_HDM_feature(test_data)
  
  create_td_feature(train_data)
  create_td_feature(test_data)
  
  create_hero_hit_features(train_data)
  create_hero_hit_features(test_data)
  
  train_features = make_coordinate_features(train_data)
  test_features = make_coordinate_features(test_data)
  
  return train_features, test_features