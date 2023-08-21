#!/usr/bin/env python3
from __future__ import print_function

import sys
sys.path.append("//media/juro/DATA/Work/Elections/code/python")

import pandas as pd

import config
import functions_sql


functions_sql.drop_table("current_data")
functions_sql.create_table_current_data()

functions_sql.drop_table("predictions")
df_parties = pd.read_csv(config.data_path + "parties.csv")
list_parties = df_parties["party_name"].to_list()
functions_sql.create_table_predictions(list_parties)


kraj_code = 1
obvod_code = 1
okres_code = 1
town_code = 1
okrsok = 1
party_ID = 1
party_name = 1
votes = 1
votes_percentage = 1
preferential_votes = 1
preferential_votes_percentage = 1


functions_sql.insert_data(kraj_code, obvod_code, okres_code, town_code, okrsok, party_ID, party_name, votes, votes_percentage, preferential_votes, preferential_votes_percentage)

X = functions_sql.select_data()

functions_sql.delete_all_data()

