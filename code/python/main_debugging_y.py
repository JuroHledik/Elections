#!/usr/bin/env python3
from __future__ import print_function

import sys
sys.path.append("//media/juro/DATA/Work/Elections/code/python")

from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import itertools
from math import isnan
import multiprocessing as mp
import numpy as np
from numpy import matlib
import os.path
import pandas as pd
from pandasql import sqldf
import scipy.sparse
import scipy.sparse.linalg

import config
import functions_formatting
import globals
import google_api_functions
import location_functions
import time

####################################################################################################################
#   DISABLE WARNINGS
####################################################################################################################
pd.options.mode.chained_assignment = None  # default='warn'

####################################################################################################################
#   GENERATE ATTENDANCE EXPECTATIONS + MOCK DATA
####################################################################################################################
# Import the data from the excel file:
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
N_df_last_elections = len(df_last_elections)

# Compute the expected attendance dataset:
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Merging data...")
df_ucast_okrsok_historical = sqldf(
    'select kraj_code, kraj, obvod_code, obvod, okres_code, okres, town_code, town, okrsok, sum(votes) as last_year_votes from df_last_elections group by kraj_code, kraj, obvod_code, obvod, okres_code, okres, town_code, town, okrsok')
df_ucast_town_historical = sqldf(
    'select kraj_code, kraj, obvod_code, obvod, okres_code, okres, town_code, town, sum(votes) as last_year_votes, count(distinct okrsok) as total_count_okrsok from df_last_elections group by kraj_code, kraj, obvod_code, obvod, okres_code, okres, town_code, town')
df_ucast_okres_historical = sqldf(
    'select kraj_code, kraj, obvod_code, obvod, okres_code, okres, sum(votes) as last_year_votes, count(distinct town_code || okrsok) as total_count_okrsok from df_last_elections group by kraj_code, kraj, obvod_code, obvod, okres_code, okres')
df_ucast_obvod_historical = sqldf(
    'select kraj_code, kraj, obvod_code, obvod, sum(votes) as last_year_votes, count(distinct obvod_code || town_code || okrsok) as total_count_okrsok from df_last_elections group by kraj_code, kraj, obvod_code, obvod')
df_ucast_kraj_historical = sqldf(
    'select kraj_code, kraj, sum(votes) as last_year_votes, count(distinct kraj_code || obvod_code || town_code || okrsok) as total_count_okrsok from df_last_elections group by kraj_code, kraj')

# #Upload the datasets into the google sheet:
# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "+ "Uploading values to the google sheet...")
# google_api_functions.upload_dataframe(df_ucast_okrsok_historical,config.RANGE_UCAST_OKRSOK)
# google_api_functions.upload_dataframe(df_ucast_town_historical,config.RANGE_UCAST_TOWN)
# google_api_functions.upload_dataframe(df_ucast_okres_historical,config.RANGE_UCAST_OKRES)
# google_api_functions.upload_dataframe(df_ucast_obvod_historical,config.RANGE_UCAST_OBVOD)
# google_api_functions.upload_dataframe(df_ucast_kraj_historical,config.RANGE_UCAST_KRAJ)

# Compute the neighbor matrix:
df_temp = df_ucast_town_historical['town_code']
A = location_functions.get_neighbor_matrix(df_temp)
# Add the diagonal to the neighbor matrix
A = A + np.eye(np.shape(A)[0])

####################################################################################################################
#   DOWNLOAD THE PARTY NUMBERS AND NAMES
####################################################################################################################
service = google_api_functions.establish_connection()

# # Call the Sheets API
result = service.spreadsheets().values().get(spreadsheetId=config.SPREADSHEET_ID,
                                             range=config.RANGE_PARTY_NAMES).execute()
values = result.get('values', [])

# Find the number of parties:
df_parties = pd.DataFrame(values, columns=['party_ID', 'party_name'])
K = len(df_parties)

globals.df_ucast_okrsok_historical = df_ucast_okrsok_historical
globals.A = A
globals.df_last_elections = df_last_elections

####################################################################################################################
#   MAIN ALGORITHM THAT RUNS THROUGH THE ELECTION NIGHT
####################################################################################################################

sigma = 0.5
low = 0.7
high = 1.3
simulation = 0
def debugging_y(sigma, low, high, simulation):
    df_ucast_okrsok_historical = globals.df_ucast_okrsok_historical
    A = globals.A
    df_last_elections = globals.df_last_elections

    np.random.seed(simulation)
    shock = np.random.uniform(low, high, len(df_ucast_okrsok_historical))
    print(shock)
    df_shock = pd.DataFrame(shock, columns=['shock'])
    df_ucast_okrsok_historical = pd.concat([df_ucast_okrsok_historical, df_shock], axis=1)
    df_shuffled_data = sqldf(
        'select a.*, b.last_year_votes as votes_per_okrsok, b.shock from df_last_elections a left join df_ucast_okrsok_historical b on a.town_code=b.town_code and a.okrsok=b.okrsok')
    df_ucast_okrsok_historical = df_ucast_okrsok_historical.drop(columns=['shock'])
    df_shuffled_data['votes_per_okrsok'] = df_shuffled_data['votes_per_okrsok'] * df_shuffled_data['shock']
    df_shuffled_data['votes'] = df_shuffled_data['votes'] * df_shuffled_data['shock']
    df_shuffled_data = df_shuffled_data.sort_values(by=['votes_per_okrsok'])

    df_shuffled_data['already_imported'] = 0
    df_shuffled_data['town_code_okrsok'] = df_shuffled_data['town_code'] + '_' + (df_shuffled_data['okrsok']).astype(str)

    # Required initiations:
    finished = False
    df_simulation = pd.DataFrame()
    temp_v_set = np.array([])
    while not finished:
        ####################################################################################################################
        #   DOWNLOAD THE DATA AND SAVE IT AS A DATAFRAME - EXCHANGE WITH SCRAPING LATER MAYBE
        ####################################################################################################################
        df_shuffled_data, df_download = formatting_functions.import_data_debugging(df_shuffled_data)

        ####################################################################################################################
        #   UPDATE DATAFRAMES OF ALL DIFFERENT GRANULARITIES AND UPLOAD THEM TO GOOGLE SHEETS
        ####################################################################################################################
        # Okrsok
        df_temp = sqldf(
            'select town_code, okrsok, sum(votes) as votes from df_download group by town_code, okrsok')
        df_ucast_okrsok = sqldf(
            'select a.*, b.votes from df_ucast_okrsok_historical a left join df_temp b on a.town_code = b.town_code and a.okrsok = b.okrsok')
        df_ucast_okrsok.loc[df_ucast_okrsok['votes'].isna(), 'votes'] = 0
        # Town
        df_temp = sqldf(
            'select town_code, sum(votes) as votes, count(distinct okrsok) as count_okrsok from df_download group by town_code')
        df_ucast_town = sqldf(
            'select a.*, b.votes, b.count_okrsok from df_ucast_town_historical a left join df_temp b on a.town_code = b.town_code')
        df_ucast_town.loc[df_ucast_town['votes'].isna(), 'votes'] = 0
        df_ucast_town.loc[df_ucast_town['count_okrsok'].isna(), 'count_okrsok'] = 0
        df_ucast_town['counted_percentage'] = df_ucast_town['count_okrsok'] / df_ucast_town[
            'total_count_okrsok']
        # # Okres
        # df_temp = sqldf(
        #     'select okres_code, sum(votes) as votes, count(distinct town_code || okrsok) as count_okrsok from df_download group by okres_code')
        # df_ucast_okres = sqldf(
        #     'select a.*, b.votes, b.count_okrsok from df_ucast_okres_historical a left join df_temp b on a.okres_code = b.okres_code')
        # df_ucast_okres.loc[df_ucast_okres['votes'].isna(), 'votes'] = 0
        # df_ucast_okres.loc[df_ucast_okres['count_okrsok'].isna(), 'count_okrsok'] = 0
        # df_ucast_okres['counted_percentage'] = df_ucast_okres['count_okrsok'] / df_ucast_okres[
        #     'total_count_okrsok']
        # # Obvod
        # df_temp = sqldf(
        #     'select obvod_code, sum(votes) as votes, count(distinct okres_code || town_code || okrsok) as count_okrsok from df_download group by obvod_code')
        # df_ucast_obvod = sqldf(
        #     'select a.*, b.votes, b.count_okrsok from df_ucast_obvod_historical a left join df_temp b on a.obvod_code = b.obvod_code')
        # df_ucast_obvod.loc[df_ucast_obvod['votes'].isna(), 'votes'] = 0
        # df_ucast_obvod.loc[df_ucast_obvod['count_okrsok'].isna(), 'count_okrsok'] = 0
        # df_ucast_obvod['counted_percentage'] = df_ucast_obvod['count_okrsok'] / df_ucast_obvod[
        #     'total_count_okrsok']
        # # Kraj
        # df_temp = sqldf(
        #     'select kraj_code, sum(votes) as votes, count(distinct obvod_code || okres_code || town_code || okrsok) as count_okrsok from df_download group by kraj_code')
        # df_ucast_kraj = sqldf(
        #     'select a.*, b.votes, b.count_okrsok from df_ucast_kraj_historical a left join df_temp b on a.kraj_code = b.kraj_code')
        # df_ucast_kraj.loc[df_ucast_kraj['votes'].isna(), 'votes'] = 0
        # df_ucast_kraj.loc[df_ucast_kraj['count_okrsok'].isna(), 'count_okrsok'] = 0
        # df_ucast_kraj['counted_percentage'] = df_ucast_kraj['count_okrsok'] / df_ucast_kraj[
        #     'total_count_okrsok']
        # Total
        counted_percentage = round(
            sqldf('select count(okrsok) from df_ucast_okrsok where votes<>0').values[0][0] / len(
                df_ucast_okrsok), 8)

        df_ucast_okrsok = sqldf(
            "select a.*, b.last_year_votes as last_year_votes_town from df_ucast_okrsok a left join df_ucast_town b on a.town_code = b.town_code")
        df_ucast_okrsok["percentage_of_town"] = df_ucast_okrsok['last_year_votes'] / df_ucast_okrsok[
            'last_year_votes_town']
        df_temp = sqldf(
            'select town_code, sum(percentage_of_town) as counted_percentage_of_votes from df_ucast_okrsok where votes!=0 group by town_code')
        df_ucast_town = sqldf(
            'select a.*, b.counted_percentage_of_votes from df_ucast_town a left join df_temp b on a.town_code=b.town_code')

        # # Upload the datasets into the google sheet:
        # print(datetime.now().strftime(
        #     '%Y-%m-%d %H:%M:%S') + ": " + "Uploading new attendance values to the google sheet...")
        # google_api_functions.upload_dataframe(df_ucast_okrsok, config.RANGE_UCAST_OKRSOK)
        # google_api_functions.upload_dataframe(df_ucast_town, config.RANGE_UCAST_TOWN)
        # # google_api_functions.upload_dataframe(df_ucast_okres, config.RANGE_UCAST_OKRES)
        # # google_api_functions.upload_dataframe(df_ucast_obvod, config.RANGE_UCAST_OBVOD)
        # # google_api_functions.upload_dataframe(df_ucast_kraj, config.RANGE_UCAST_KRAJ)

        # ####################################################################################################################
        # #   UPDATE THE GOOGLE SHEET WITH NEW DATA
        # ####################################################################################################################
        # # Update the new data with the timestamp and counted_percentage variables:
        # df_sheet_update = sqldf(
        #     'select b.timestamp, b.finished_percentage, a.* from df_download a left join df_current_data b on a.town_code=b.town_code and a.okrsok=b.okrsok and a.party_ID=b.party_ID ')
        # df_sheet_update = df_sheet_update[df_sheet_update["timestamp"].isna()]
        # data_update_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # df_sheet_update.loc[df_sheet_update['timestamp'].isna(), 'timestamp'] = data_update_timestamp
        # df_sheet_update.loc[
        #     df_sheet_update['finished_percentage'].isna(), 'finished_percentage'] = counted_percentage
        #
        # google_api_functions.append_dataframe(df_sheet_update, config.RANGE_GRANULAR_DATA_WRITE)
        # google_api_functions.upload_dataframe(pd.DataFrame([data_update_timestamp]),
        #                                       config.RANGE_DASHBOARD_LAST_DATA_UPDATE)

        ####################################################################################################################
        #   COMPUTE THE EXPECTED VOTER ATTENDANCE
        ####################################################################################################################
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Updating expected attendance...")

        # Set the variables used in computation
        df_ucast_computation = df_ucast_town[
            ['town_code', 'last_year_votes', 'votes', 'counted_percentage', 'counted_percentage_of_votes']]
        df_ucast_computation.loc[df_ucast_computation['counted_percentage'].isna(), 'votes'] = 0
        df_ucast_computation.loc[
            df_ucast_computation['counted_percentage_of_votes'].isna(), 'counted_percentage_of_votes'] = 0
        df_ucast_computation.loc[
            df_ucast_computation['counted_percentage'].isna(), 'counted_percentage'] = config.eps

        v_hat = df_ucast_computation['last_year_votes'].values
        v_bar = df_ucast_computation['votes'].values
        c_bar = df_ucast_computation['counted_percentage_of_votes'].values
        N = np.shape(A)[0]

        # Set the tilde variables, see documentation
        v_tilde = v_bar / v_hat
        A_tilde = (np.diag((1 - c_bar) / A.dot(np.ones(N)))).dot(A)

        # Set the theta:
        temp_expected_ucast = np.sum(np.sign(c_bar) * v_tilde / (c_bar + config.eps)) / np.sum(np.sign(c_bar))
        lower_bar = 0.95 * (1 - counted_percentage) + 0.55 * counted_percentage
        upper_bar = 1.2 * (1 - counted_percentage) + 1.05 * counted_percentage
        if temp_expected_ucast < lower_bar:
            theta = 0.95
        else:
            if temp_expected_ucast < 1:
                theta = min((temp_expected_ucast - lower_bar) / (1 - lower_bar) * 0.95, 0.95)
            else:
                if temp_expected_ucast < upper_bar:
                    theta = min((upper_bar - temp_expected_ucast) / (upper_bar - 1), 0.95)
                else:
                    theta = 0.95

        # Set the final variables that go into the linear solver
        AA = scipy.sparse.csr_matrix(np.eye(N) - theta * A_tilde)
        # bb = v_tilde + (1 - theta) * (1 - c_bar) * (np.sign(c_bar) * (v_tilde / (c_bar + config.eps)) + (
        #             1 - np.sign(c_bar)) * temp_expected_ucast)
        bb = v_tilde + (1 - theta) * (1 - c_bar)

        # Solve the system and compute the expected number of votes per town
        phi = scipy.sparse.linalg.spsolve(AA, bb)
        v = phi * v_hat

        df_v = pd.DataFrame(v, columns=['expected_votes'])
        df_ucast_computation = pd.concat([df_ucast_computation, df_v], axis=1)

        ####################################################################################################################
        #  COMPUTE THE EXPECTED VOTES PER PARTY
        ####################################################################################################################
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Computing expected votes per party...")
        # Town dataset containing the party_ID and votes
        df_parties_town = sqldf(
            'select town_code, party_ID, sum(votes) as votes from df_download group by town_code, party_ID')
        # y_star variables will be stacked here:
        y_star_set = np.array([])
        current_votes_per_party = []

        A_tilde = (np.diag((1 - c_bar) / A.dot(v))).dot(np.multiply(A, np.tile(v, (N, 1))))
        AA = scipy.sparse.csr_matrix(np.eye(N) - sigma * A_tilde)

        party_ID_index = 10
        for party_ID_index in range(0, K):
            party_ID = df_parties['party_ID'].iloc[party_ID_index]
            # Only select one party, make sure indexing is correct by using left join to base table df_ucast_town_historical
            df_temp = sqldf('select * from df_parties_town where party_ID=' + party_ID)
            df_party_town = sqldf(
                'select a.*, b.votes from df_ucast_town_historical a left join df_temp b on a.town_code = b.town_code')

            df_party_computation = df_party_town[['town_code', 'votes']]
            df_party_computation.loc[df_party_computation['votes'].isna(), 'votes'] = 0

            y_bar = df_party_computation['votes'].values / (v_bar + config.eps)

            # If y_bar is zero, so if this party has absolutely no votes at all, automatically put y_star = 0 as well.
            # Otherwise we get an error in the sparse linear solver for some reason.
            if sum(y_bar) == 0:
                y_star = y_bar
            else:
                bb = c_bar * y_bar + (1 - sigma) * (1 - c_bar) * y_bar

                # Solve the system and compute the expected number of votes per town
                y_star = scipy.sparse.linalg.spsolve(AA, bb)

            current_votes = sum(np.transpose(y_bar * v_bar))

            # Stack the vectors:
            y_star_set = np.vstack([y_star_set, y_star]) if y_star_set.size else y_star
            current_votes_per_party.append(current_votes)

        sum_y_star_set = sum(y_star_set)
        sum_y_star_set[sum_y_star_set == 0] = 1

        y = y_star_set / (np.tile(sum_y_star_set, (K, 1)))

        expected_votes_per_party = sum(np.transpose(y * (np.tile(v, (K, 1)))))

        prediction_update_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        expected_results = expected_votes_per_party / sum(expected_votes_per_party)
        temp = list(expected_results)
        temp.insert(0, counted_percentage)
        temp.insert(0, prediction_update_timestamp)
        df_expected_results = pd.DataFrame(temp)

        current_results = current_votes_per_party / sum(current_votes_per_party)
        temp = list(current_results)
        temp.insert(0, counted_percentage)
        temp.insert(0, prediction_update_timestamp)
        df_current_results = pd.DataFrame(temp)

        # ####################################################################################################################
        # #  UPDATE THE GOOGLE SHEET WITH THE LATEST PREDICTION
        # ####################################################################################################################
        # print(datetime.now().strftime(
        #     '%Y-%m-%d %H:%M:%S') + ": " + "Updating the google sheet with the new prediction...")
        # google_api_functions.append_dataframe(df_expected_results.transpose(), config.RANGE_PREDICTION)
        # google_api_functions.append_dataframe(df_current_results.transpose(), config.RANGE_CURRENT_RESULTS)
        #
        # google_api_functions.upload_dataframe(df_current_results, config.RANGE_DASHBOARD_CURRENT_RESULTS)
        # df_temp_prediction = google_api_functions.get_dataframe(config.RANGE_DASHBOARD_CURRENT_PREDICTION)
        # google_api_functions.upload_dataframe(df_temp_prediction, config.RANGE_DASHBOARD_LAST_PREDICTION)
        # google_api_functions.upload_dataframe(df_expected_results, config.RANGE_DASHBOARD_CURRENT_PREDICTION)
        #
        # google_api_functions.upload_dataframe(pd.DataFrame([prediction_update_timestamp]),
        #                                       config.RANGE_DASHBOARD_LAST_PREDICTION_UPDATE)

        ####################################################################################################################
        #  FINISH THE CYCLE
        ####################################################################################################################
        # temp_v_set = np.vstack([temp_v_set, v]) if temp_v_set.size else v
        df_simulation = df_simulation.append(pd.DataFrame(expected_results).transpose())

        if df_ucast_town['counted_percentage'].mean() == 1:
            finished = 1
        print(datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + ": " + "Cycle finished. Currently accounted for " + str(
            100 * counted_percentage) + " % of voting places.")

    # Download the prediction evolution and save it as a csv file:
    df_simulation.to_csv(config.data_path + "simulation_y/y" + "_sigma_" + str(sigma) + "_low_" + str(low) + "_high_" + str(high) + "_simulation_" + str(simulation) +".csv", index=False)
    # np.savetxt(config.data_path + "simulation/v" + "_theta_" + str(theta) + "_gamma_" + str(gamma) + "_sigma_" + str(sigma) + "_low_" + str(low) + "_high_" + str(high) + ".csv", sim_results, delimiter=",")
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Simulation completed.")

# sigma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# low = [0.7,0.9]
# high = [1.1,1.3]

simulation = range(12,150)
sigma = [0.5]
low = [0.7]
high = [1.3]

pool = mp.Pool(mp.cpu_count() - 1)

parameter_combinations = list(itertools.product(sigma, low, high, simulation))
pool.starmap_async(debugging_y, parameter_combinations).get()

pool.close()
pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

