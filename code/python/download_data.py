#!/usr/bin/env python3
from __future__ import print_function

import sys
sys.path.append("//media/juro/DATA/Work/Elections/code/python")

from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import json
from math import isnan
import numpy as np
from numpy import matlib
import os.path
import pandas as pd
from pandasql import sqldf
import scipy.sparse
import scipy.sparse.linalg
import shutil
import urllib.request

import config
import functions_formatting
import functions_sql
import google_api_functions
import location_functions
import time

####################################################################################################################
#   DELETE all data in the MySQL database
####################################################################################################################
functions_sql.delete_all_data()

####################################################################################################################
#   GET IDs OF GEO CLASSES (KRAJ, OBVOD, OKRES, ...)
####################################################################################################################
# Import the data from the excel file:
df_vote_count = functions_formatting.import_okrsok_list()

# Create a database of geo ID combinations:
df_vote_count = sqldf('select distinct kraj_code, obvod_code, okres_code, town_code, okrsok from df_vote_count')
df_vote_count["votes"] = 0

####################################################################################################################
#   GET PARTY DATA:
####################################################################################################################
df_parties = pd.read_csv(config.data_path + "parties.csv")
df_parties.columns = ['party_ID', 'party_name']
K = len(df_parties)
party_id_set = df_parties["party_ID"].values

####################################################################################################################
#   CHECK WHETHER THERE IS NEW DATA:
####################################################################################################################

# While there are places with zero votes, try to download new data:
added_okrsok_so_far = 0
while df_vote_count["votes"].min() == 0:
    # Load the currently saved data and update the vote count for each okrsok:
    df_current_data = functions_formatting.import_current_data()
    df_current_data = df_current_data.drop_duplicates()
    df_current_data_agg = sqldf("select kraj_code, obvod_code, okres_code, town_code, okrsok, sum(votes) as votes from df_current_data group by kraj_code, obvod_code, okres_code, town_code, okrsok")
    df_vote_count = pd.concat([df_vote_count, df_current_data_agg], axis=0)
    df_vote_count = sqldf("select kraj_code, obvod_code, okres_code, town_code, okrsok, max(votes) as votes from df_vote_count group by kraj_code, obvod_code, okres_code, town_code, okrsok")
    old_vote_count = df_vote_count["votes"].sum()

    # Download and open the main json file:
    new_data = functions_formatting.download_main_json_file()
    new_vote_count = new_data["votes"].sum()

    kraj = '1'
    obvod = '101'
    okres = '101'
    town = '528595'
    okrsok = 1
    # If the new vote count is higher than the old, add the new data:
    if new_vote_count != old_vote_count:
        for kraj in df_vote_count["kraj_code"].unique():
            temp_df_kraj = df_vote_count[df_vote_count['kraj_code'] == kraj]
            old_vote_count_kraj = temp_df_kraj["votes"].sum()
            try:
                #Check for a new data in this kraj:
                new_data_kraj = functions_formatting.download_kraj_json_file(kraj)
                new_vote_count_kraj = new_data_kraj["votes"].sum()
            except:
                # If there is no data in this kraj:
                new_vote_count_kraj = old_vote_count_kraj

            if new_vote_count_kraj != old_vote_count_kraj:
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "New values for kraj_code = " + str(kraj))
                for obvod in df_vote_count[(df_vote_count['kraj_code'] == kraj)]["obvod_code"].unique():
                    temp_df_obvod = df_vote_count[(df_vote_count['kraj_code'] == kraj) & (df_vote_count["obvod_code"] == obvod)]
                    old_vote_count_obvod = temp_df_obvod["votes"].sum()
                    try:
                        # Check for a new data in this obvod:
                        new_data_obvod = functions_formatting.download_obvod_json_file(kraj, obvod)
                        new_vote_count_obvod = new_data_obvod["votes"].sum()
                    except:
                        # If there is no data in this obvod:
                        new_vote_count_obvod = old_vote_count_obvod

                    if new_vote_count_obvod != old_vote_count_obvod:
                        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "New values for obvod_code = " + str(obvod))
                        for okres in temp_df_obvod["okres_code"].unique():
                            temp_df_okres = df_vote_count[(df_vote_count['kraj_code'] == kraj) & (df_vote_count["obvod_code"] == obvod) & (df_vote_count["okres_code"] == okres)]
                            old_vote_count_okres = temp_df_okres["votes"].sum()
                            try:
                                # Check for a new data in this okres:
                                new_data_okres = functions_formatting.download_okres_json_file(kraj, obvod, okres)
                                new_vote_count_okres = new_data_okres["votes"].sum()
                            except:
                                # If there is no data in this okres:
                                new_vote_count_okres = old_vote_count_okres

                            if new_vote_count_okres != old_vote_count_okres:
                                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "New values for okres_code = " + str(okres))
                                for town in temp_df_okres["town_code"].unique():
                                    temp_df_town = df_vote_count[(df_vote_count['kraj_code'] == kraj) & (df_vote_count["obvod_code"] == obvod) & (df_vote_count["okres_code"] == okres) & (df_vote_count["town_code"] == town)]
                                    old_vote_count_town = temp_df_town["votes"].sum()
                                    try:
                                        # Check for a new data in this town:
                                        new_data_town = functions_formatting.download_town_json_file(kraj, obvod, okres, town)
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
                                                new_data_okrsok = functions_formatting.download_okrsok_json_file(kraj, obvod, okres, town, okrsok)
                                                new_vote_count_okrsok = new_data_okrsok["votes"].sum()
                                            except:
                                                # If there is no data in this okrsok:
                                                new_vote_count_okrsok = old_vote_count_okrsok

                                            if new_vote_count_okrsok != old_vote_count_okrsok:
                                                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Adding values for kraj = " + str(kraj) + ", obvod = " + str(obvod) + ", okres = " + str(okres) + ", town = " + str(town) + ", okrsok = " + str(okrsok))

                                                # If a new okrsok is counted add the observations to the database:
                                                for new_data_okrsok_row_index in range(0, len(new_data_okrsok)):
                                                    timestamp = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                                                    finished_percentage = str(added_okrsok_so_far/len(df_vote_count))
                                                    kraj_code = int(new_data_okrsok["kraj_code"][new_data_okrsok_row_index])
                                                    obvod_code = int(new_data_okrsok["obvod_code"][new_data_okrsok_row_index])
                                                    okres_code = int(new_data_okrsok["okres_code"][new_data_okrsok_row_index])
                                                    town_code = int(new_data_okrsok["town_code"][new_data_okrsok_row_index])
                                                    okrsok = int(new_data_okrsok["okrsok"][new_data_okrsok_row_index])
                                                    party_ID = int(new_data_okrsok["party_ID"][new_data_okrsok_row_index])
                                                    party_name = str(new_data_okrsok["party_name"][new_data_okrsok_row_index])
                                                    votes = int(new_data_okrsok["votes"][new_data_okrsok_row_index])
                                                    votes_percentage = str(new_data_okrsok["votes_percentage"][new_data_okrsok_row_index])
                                                    preferential_votes = str(new_data_okrsok["preferential_votes"][new_data_okrsok_row_index])
                                                    preferential_votes_percentage = str(new_data_okrsok["preferential_votes_percentage"][new_data_okrsok_row_index])
                                                    functions_sql.insert_data(timestamp, finished_percentage, kraj_code, obvod_code, okres_code, town_code, okrsok, party_ID, party_name, votes, votes_percentage, preferential_votes, preferential_votes_percentage)
                                                added_okrsok_so_far = added_okrsok_so_far + 1
                            print("Added okrsok so far: " + str(added_okrsok_so_far))







