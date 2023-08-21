import config
import copy
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
from pandasql import sqldf
import urllib.request

def clear_current_data(path):
    f = open(path, "w")
    f.truncate()
    f.close()

def create_directory(path):
    try:
        os.mkdir(path)
    except OSError:
        pass
    else:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') +": Successfully created the directory %s " % path)

def import_data_debugging(df_shuffled_data):
    distinct_town_code_okrsok = sqldf('select distinct town_code_okrsok from df_shuffled_data where already_imported=0 limit ' + str(config.NumberOfOkrsokPerIteration))['town_code_okrsok'].values
    df_shuffled_data.loc[df_shuffled_data['town_code_okrsok'].isin(distinct_town_code_okrsok), 'already_imported'] = 1
    df_download = sqldf('select kraj_code, obvod_code, okres_code, town_code, okrsok, party_ID, party_name, votes, votes_percentage, preferential_votes, preferential_votes_percentage from df_shuffled_data where already_imported=1 order by kraj_code, obvod_code, okres_code, town_code, okrsok, party_ID')
    return df_shuffled_data, df_download

def import_okrsok_list():
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Importing the okrsok list...")
    df_uzemne_clenenie = pd.read_excel(config.data_path + "uzemne_clenenie_" + str(config.year) + ".xlsx")
    df_uzemne_clenenie.columns = ['kraj_code',
                                 'kraj',
                                 'obvod_code',
                                 'obvod',
                                 'okres_code',
                                 'okres',
                                 'town_code',
                                 'town',
                                 'okrsok']
    df_uzemne_clenenie = df_uzemne_clenenie.iloc[2:len(df_uzemne_clenenie), ]
    df_uzemne_clenenie = df_uzemne_clenenie.dropna(how='any', axis=0)
    df_uzemne_clenenie = df_uzemne_clenenie.reset_index()
    df_uzemne_clenenie = df_uzemne_clenenie.drop(columns=['index'])
    df_okrsok_list = df_uzemne_clenenie[0:0]
    for i in range(0,len(df_uzemne_clenenie)):
        for j in range(0,int((df_uzemne_clenenie["okrsok"][i]))):
            temp = pd.DataFrame({"kraj_code":[df_uzemne_clenenie["kraj_code"][i]],
                                 "kraj": [df_uzemne_clenenie["kraj"][i]],
                                 "obvod_code": [df_uzemne_clenenie["obvod_code"][i]],
                                 "obvod": [df_uzemne_clenenie["obvod"][i]],
                                 "okres_code": [df_uzemne_clenenie["okres_code"][i]],
                                 "okres": [df_uzemne_clenenie["okres"][i]],
                                 "town_code": [df_uzemne_clenenie["town_code"][i]],
                                 "town": [df_uzemne_clenenie["town"][i]],
                                 "okrsok": str(j+1)})
            df_okrsok_list = df_okrsok_list.append(temp)

    df_okrsok_list = df_okrsok_list.reset_index(drop=True)

    return df_okrsok_list


def import_historical_data():
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Importing the historical data...")
    df_last_elections = pd.read_excel(config.data_path + "historical_data.xlsx")
    df_last_elections.columns = ['kraj_code',
                                 'kraj',
                                 'obvod_code',
                                 'obvod',
                                 'okres_code',
                                 'okres',
                                 'town_code',
                                 'town',
                                 'okrsok',
                                 'party_ID',
                                 'party_name',
                                 'votes',
                                 'votes_percentage',
                                 'preferential_votes',
                                 'preferential_votes_percentage']
    df_last_elections = df_last_elections.iloc[2:len(df_last_elections), ]
    df_last_elections = df_last_elections.dropna(how='any', axis=0)
    df_last_elections = df_last_elections.reset_index()
    del df_last_elections['index']
    return df_last_elections

def import_current_data():
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Importing the current data...")
    df_current_data = pd.read_csv(config.data_path + "current_data.csv", header=0) #First row is taken as header - row 0
    columns = ['kraj_code',
               'obvod_code',
               'okres_code',
               'town_code',
               'okrsok',
               'party_ID',
               'party_name',
               'votes',
               'votes_percentage',
               'preferential_votes',
               'preferential_votes_percentage']
    if df_current_data.shape[0] == 0:
        df_current_data = pd.DataFrame(columns=columns)
    else:
        df_current_data.columns = columns

    df_current_data = df_current_data.reset_index()
    df_current_data["votes"] = df_current_data["votes"].astype(int)
    del df_current_data['index']
    return df_current_data

def download_main_json_file():
    create_directory(config.data_path + 'web_structure')
    create_directory(config.data_path + 'web_structure/json')
    file_path = config.data_path + 'web_structure/json/' + 'tab03a.json'
    urllib.request.urlretrieve(config.web_path + "json/tab03a.json", file_path)
    # Opening JSON file
    new_data = pd.DataFrame(json.load(open(file_path)))
    new_data.columns = ["party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage", "mandates"]
    new_data["votes"] = new_data["votes"].str.replace(" ","").astype(int)
    new_data["party_ID"] = new_data["party_ID"].str.replace(" ", "").astype(int)
    return new_data

def download_kraj_json_file(kraj):
    create_directory(config.data_path + 'web_structure')
    create_directory(config.data_path + 'web_structure/json')
    create_directory(config.data_path + 'web_structure/json/tab03b')
    file_path = config.data_path + 'web_structure/json/tab03b/' + str(kraj) + ".json"
    urllib.request.urlretrieve(config.web_path + "json/tab03b/" + str(kraj) + ".json", file_path)
    # Opening JSON file
    new_data_kraj = pd.DataFrame(json.load(open(file_path)))
    new_data_kraj.columns = ["party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]
    new_data_kraj["votes"] = new_data_kraj["votes"].str.replace(" ","").astype(int)
    new_data_kraj["party_ID"] = new_data_kraj["party_ID"].str.replace(" ", "").astype(int)
    new_data_kraj["kraj_code"] = kraj
    new_data_kraj = new_data_kraj[["kraj_code", "party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]]
    return new_data_kraj

def download_obvod_json_file(kraj, obvod):
    create_directory(config.data_path + 'web_structure')
    create_directory(config.data_path + 'web_structure/json')
    create_directory(config.data_path + 'web_structure/json/tab03c')
    file_path = config.data_path + 'web_structure/json/tab03c/' + str(obvod) + ".json"
    urllib.request.urlretrieve(config.web_path + "json/tab03c/" + str(obvod) + ".json", file_path)
    # Opening JSON file
    new_data_obvod = pd.DataFrame(json.load(open(file_path)))
    new_data_obvod.columns = ["party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]
    new_data_obvod["votes"] = new_data_obvod["votes"].str.replace(" ","").astype(int)
    new_data_obvod["party_ID"] = new_data_obvod["party_ID"].str.replace(" ", "").astype(int)
    new_data_obvod["kraj_code"] = kraj
    new_data_obvod["obvod_code"] = obvod
    new_data_obvod = new_data_obvod[["kraj_code", "obvod_code", "party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]]
    return new_data_obvod

def download_okres_json_file(kraj, obvod, okres):
    create_directory(config.data_path + 'web_structure')
    create_directory(config.data_path + 'web_structure/json')
    create_directory(config.data_path + 'web_structure/json/tab03d')
    file_path = config.data_path + 'web_structure/json/tab03d/' + str(okres) + ".json"
    urllib.request.urlretrieve(config.web_path + "json/tab03d/" + str(okres) + ".json", file_path)
    # Opening JSON file
    new_data_okres = pd.DataFrame(json.load(open(file_path)))
    new_data_okres.columns = ["party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]
    new_data_okres["votes"] = new_data_okres["votes"].str.replace(" ","").astype(int)
    new_data_okres["party_ID"] = new_data_okres["party_ID"].str.replace(" ", "").astype(int)
    new_data_okres["kraj_code"] = kraj
    new_data_okres["obvod_code"] = obvod
    new_data_okres["okres_code"] = okres
    new_data_okres = new_data_okres[["kraj_code", "obvod_code", "okres_code", "party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]]
    return new_data_okres

def download_town_json_file(kraj, obvod, okres, town):
    create_directory(config.data_path + 'web_structure')
    create_directory(config.data_path + 'web_structure/json')
    create_directory(config.data_path + 'web_structure/json/tab03e')
    file_path = config.data_path + 'web_structure/json/tab03e/' + str(town) + ".json"
    urllib.request.urlretrieve(config.web_path + "json/tab03e/" + str(town) + ".json", file_path)
    # Opening JSON file
    new_data_town = pd.DataFrame(json.load(open(file_path)))
    new_data_town.columns = ["party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]
    new_data_town["votes"] = new_data_town["votes"].str.replace(" ","").astype(int)
    new_data_town["party_ID"] = new_data_town["party_ID"].str.replace(" ", "").astype(int)
    new_data_town["kraj_code"] = kraj
    new_data_town["obvod_code"] = obvod
    new_data_town["okres_code"] = okres
    new_data_town["town_code"] = town
    new_data_town = new_data_town[["kraj_code", "obvod_code", "okres_code", "town_code", "party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]]
    return new_data_town

def download_okrsok_json_file(kraj, obvod, okres, town, okrsok):
    create_directory(config.data_path + 'web_structure')
    create_directory(config.data_path + 'web_structure/json')
    create_directory(config.data_path + 'web_structure/json/tab03f')
    file_path = config.data_path + 'web_structure/json/tab03f/' + str(town) + str(okrsok) + ".json"
    urllib.request.urlretrieve(config.web_path + "json/tab03f/" + str(town) + str(okrsok) + ".json", file_path)
    # Opening JSON file
    new_data_okrsok = pd.DataFrame(json.load(open(file_path)))
    new_data_okrsok.columns = ["party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]
    new_data_okrsok["votes"] = new_data_okrsok["votes"].str.replace(" ","").astype(int)
    new_data_okrsok["party_ID"] = new_data_okrsok["party_ID"].str.replace(" ", "").astype(int)
    new_data_okrsok["kraj_code"] = kraj
    new_data_okrsok["obvod_code"] = obvod
    new_data_okrsok["okres_code"] = okres
    new_data_okrsok["town_code"] = town
    new_data_okrsok["okrsok"] = okrsok
    new_data_okrsok = new_data_okrsok[["kraj_code", "obvod_code", "okres_code", "town_code", "okrsok", "party_ID", "party_name", "votes", "votes_percentage", "preferential_votes", "preferential_votes_percentage"]]
    return new_data_okrsok


# Parallel download of town data
def download_parallel_town(df_vote_count, kraj, obvod, okres, town):
    added_okrsok_per_town = 0
    temp_df_town = df_vote_count[(df_vote_count['kraj_code'] == kraj) & (df_vote_count["obvod_code"] == obvod) & (df_vote_count["okres_code"] == okres) & (df_vote_count["town_code"] == town)]
    old_vote_count_town = temp_df_town["votes"].sum()
    try:
        # Check for a new data in this town:
        new_data_town = download_town_json_file(kraj, obvod, okres, town)
        new_vote_count_town = new_data_town["votes"].sum()
    except:
        # If there is no data in this town:
        new_vote_count_town = old_vote_count_town

    if new_vote_count_town != old_vote_count_town:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "New values for town_code = " + str(town))
        for okrsok in df_vote_count[(df_vote_count["town_code"] == town) & (df_vote_count["votes"] == 0)]["okrsok"].unique():
            temp_df_okrsok = df_vote_count[(df_vote_count['kraj_code'] == kraj) & (df_vote_count["obvod_code"] == obvod) & (df_vote_count["okres_code"] == okres) & (df_vote_count["town_code"] == town) & (df_vote_count["okrsok"] == okrsok)]
            old_vote_count_okrsok = temp_df_okrsok["votes"].sum()
            try:
                # Check for a new data in this okrsok:
                new_data_okrsok = download_okrsok_json_file(kraj, obvod, okres, town, okrsok)
                new_vote_count_okrsok = new_data_okrsok["votes"].sum()
            except:
                # If there is no data in this okrsok:
                new_vote_count_okrsok = old_vote_count_okrsok

            if new_vote_count_okrsok != old_vote_count_okrsok:
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Adding values for kraj = " + str(kraj) + ", obvod = " + str(obvod) + ", okres = " + str(okres) + ", town = " + str(town) + ", okrsok = " + str(okrsok))
                added_okrsok_per_town = added_okrsok_per_town + 1
                return new_data_okrsok, added_okrsok_per_town


def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")