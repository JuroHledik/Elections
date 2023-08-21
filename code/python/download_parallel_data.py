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
import multiprocessing
import os.path
import pandas as pd
from pandasql import sqldf
import scipy.sparse
import scipy.sparse.linalg
import shutil
import urllib.request

import config
import functions_formatting
import google_api_functions
import location_functions
import time

####################################################################################################################
#   DELETE data.csv if you want to start fresh
####################################################################################################################
# Delete the current_data.csv file and create it again from current_data_backup.csv
try:
    os.remove(config.data_path + "current_data.csv")
except:
    print("Cannot remove current_data.csv or copy the backup file")
shutil.copy(config.data_path + "current_data_backup.csv", config.data_path + "current_data.csv")

####################################################################################################################
#   GET IDs OF GEO CLASSES (KRAJ, OBVOD, OKRES, ...)
####################################################################################################################
# Import the data from the excel file:
df_vote_count = formatting_functions.import_okrsok_list()

# Create a database of geo ID combinations:
df_vote_count = sqldf('select distinct kraj_code, obvod_code, okres_code, town_code, okrsok from df_vote_count')
df_vote_count["votes"] = 0

####################################################################################################################
#   GET PARTY DATA:
####################################################################################################################
df_parties = google_api_functions.get_dataframe(config.RANGE_PARTY_NAMES, config.SPREADSHEET_ID)
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
    df_current_data = formatting_functions.import_current_data()
    df_current_data = df_current_data.drop_duplicates()
    df_current_data_agg = sqldf("select kraj_code, obvod_code, okres_code, town_code, okrsok, sum(votes) as votes from df_current_data group by kraj_code, obvod_code, okres_code, town_code, okrsok")
    df_vote_count = pd.concat([df_vote_count, df_current_data_agg], axis=0)
    df_vote_count = sqldf("select kraj_code, obvod_code, okres_code, town_code, okrsok, max(votes) as votes from df_vote_count group by kraj_code, obvod_code, okres_code, town_code, okrsok")
    old_vote_count = df_vote_count["votes"].sum()

    # Download and open the main json file:
    new_data = formatting_functions.download_main_json_file()
    new_vote_count = new_data["votes"].sum()

    # kraj = '1'
    # obvod = '101'
    # okres = '101'
    # town = '528595'
    # okrsok = 1
    # If the new vote count is higher than the old, add the new data:
    if new_vote_count != old_vote_count:
        for kraj in df_vote_count["kraj_code"].unique():
            temp_df_kraj = df_vote_count[df_vote_count['kraj_code'] == kraj]
            old_vote_count_kraj = temp_df_kraj["votes"].sum()
            try:
                #Check for a new data in this kraj:
                new_data_kraj = formatting_functions.download_kraj_json_file(kraj)
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
                        new_data_obvod = formatting_functions.download_obvod_json_file(kraj, obvod)
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
                                new_data_okres = formatting_functions.download_okres_json_file(kraj, obvod, okres)
                                new_vote_count_okres = new_data_okres["votes"].sum()
                            except:
                                # If there is no data in this okres:
                                new_vote_count_okres = old_vote_count_okres

                            if new_vote_count_okres != old_vote_count_okres:
                                with multiprocessing.Pool() as pool:
                                    items = []
                                    for town in temp_df_okres["town_code"].unique():
                                        items.append((df_vote_count, kraj, obvod, okres, town))
                                    # call the function for each item in parallel
                                    for new_data_okrsok, result_added_okrsok_per_town in pool.starmap(formatting_functions.download_parallel_town, items):
                                        df_current_data = pd.concat([df_current_data, new_data_okrsok], axis=0)
                                        added_okrsok_so_far = added_okrsok_so_far + result_added_okrsok_per_town

                            #After each OKRES has been scanned, save the currently added data in the designated excel file
                            path = config.data_path + "current_data.csv"
                            formatting_functions.clear_current_data(path)
                            df_current_data.to_csv(path, header=True, index=False)
                            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Added the following number of okrsok so far: " + str(added_okrsok_so_far))






