#!/usr/bin/env python3
from __future__ import print_function

import sys
# sys.path.append("//media/juro/DATA/Work/Elections/code/python")
sys.path.append("//mnt/b37936f4-84ca-4659-b59a-3142ded0de7c/Projects/Elections/code/python")

from datetime import datetime

from math import isnan
import numpy as np
from numpy import matlib
import os.path
import pandas as pd
from pandasql import sqldf
import scipy.sparse
import scipy.sparse.linalg
from scipy.stats import norm
import shutil

import config
import functions_formatting
# import functions_sql

import location_functions
import time

####################################################################################################################
#   DISABLE WARNINGS ABOUT CHAINED ASSIGNMENT
####################################################################################################################
pd.options.mode.chained_assignment = None  # default='warn'

####################################################################################################################
#   CLEAR THE DATABASE
####################################################################################################################
# # Initiate an empty table for the predictions:
# functions_sql.drop_table("predictions")
# df_parties = pd.read_csv(config.data_path + "parties.csv")
# list_parties = df_parties["party_name"].to_list()
# functions_sql.create_table_predictions(list_parties)

# Delete all files in data/gui:
directory_path = config.data_path + '/gui'
functions_formatting.delete_files_in_directory(directory_path)

####################################################################################################################
#   GENERATE ATTENDANCE EXPECTATIONS + MOCK DATA
####################################################################################################################
#Import the data from the excel file:
df_last_elections = functions_formatting.import_historical_data()
N_df_last_elections = len(df_last_elections)

#Compute the expected attendance dataset:
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "+ "Merging data...")
df_ucast_okrsok_historical = sqldf('select kraj_code, kraj, obvod_code, obvod, okres_code, okres, town_code, town, okrsok, sum(votes) as last_year_votes from df_last_elections group by kraj_code, kraj, obvod_code, obvod, okres_code, okres, town_code, town, okrsok')
df_ucast_town_historical = sqldf('select kraj_code, kraj, obvod_code, obvod, okres_code, okres, town_code, town, sum(votes) as last_year_votes, count(distinct okrsok) as total_count_okrsok from df_last_elections group by kraj_code, kraj, obvod_code, obvod, okres_code, okres, town_code, town')
df_ucast_okres_historical = sqldf('select kraj_code, kraj, obvod_code, obvod, okres_code, okres, sum(votes) as last_year_votes, count(distinct town_code || okrsok) as total_count_okrsok from df_last_elections group by kraj_code, kraj, obvod_code, obvod, okres_code, okres')
df_ucast_obvod_historical = sqldf('select kraj_code, kraj, obvod_code, obvod, sum(votes) as last_year_votes, count(distinct obvod_code || town_code || okrsok) as total_count_okrsok from df_last_elections group by kraj_code, kraj, obvod_code, obvod')
df_ucast_kraj_historical = sqldf('select kraj_code, kraj, sum(votes) as last_year_votes, count(distinct kraj_code || obvod_code || town_code || okrsok) as total_count_okrsok from df_last_elections group by kraj_code, kraj')

#Compute the neighbor matrix:
df_temp = df_ucast_town_historical['town_code']
A = location_functions.get_neighbor_matrix(df_temp)
#Add the diagonal to the neighbor matrix
A = A + np.eye(np.shape(A)[0])

####################################################################################################################
#   SET THE PARTY NUMBERS AND NAMES
####################################################################################################################
df_parties = pd.read_csv(config.data_path + "parties.csv")
df_parties.columns = ['party_ID', 'party_name']
K = len(df_parties)

####################################################################################################################
#   DOWNLOAD THE PREDICTION QUANTILES
####################################################################################################################
df_quantiles = pd.read_csv(config.data_path + "quantiles.csv")
df_quantiles.columns = ['counted_percentage', 'lower_quantile', 'upper_quantile', 'mu', 'sigma_squared']
df_quantiles = df_quantiles.astype(float)

####################################################################################################################
#   SET UP THE CURRENT DATA FILE
####################################################################################################################
# Find the number of observations already processed. We don't want to simply put N_old = 0, because this way if anything
# goes wrong, we can continue from our most recent state by re-running this script.
# df_current_data = functions_sql.select_data()
# N_old = df_current_data["votes"].sum()

#Only for JRC debugging
N_old = 0

####################################################################################################################
#   MAIN ALGORITHM THAT RUNS THROUGH THE ELECTION NIGHT
####################################################################################################################
# Artificially generate shuffled data that is used to incrementally add more okrsok. This is only for simulations, not for an actual election night:
shock = np.random.uniform(0.8,1.2,len(df_ucast_okrsok_historical))
df_shock = pd.DataFrame(shock, columns=['shock'])
df_ucast_okrsok_historical = pd.concat([df_ucast_okrsok_historical, df_shock], axis=1)
df_shuffled_data = sqldf('select a.*, b.last_year_votes as votes_per_okrsok, b.shock from df_last_elections a left join df_ucast_okrsok_historical b on a.town_code=b.town_code and a.okrsok=b.okrsok')
df_ucast_okrsok_historical = df_ucast_okrsok_historical.drop(columns=['shock'])
df_shuffled_data['votes_per_okrsok'] = df_shuffled_data['votes_per_okrsok']*df_shuffled_data['shock']
df_shuffled_data['votes'] = df_shuffled_data['votes']*df_shuffled_data['shock']
df_shuffled_data = df_shuffled_data.sort_values(by=['votes_per_okrsok'])
df_shuffled_data['already_imported'] = 0
df_shuffled_data['town_code_okrsok'] = df_shuffled_data['town_code'] + '_' + (df_shuffled_data['okrsok']).astype(str)

#Required initiations:
finished = False
theta = 0.9

while not finished:
    ####################################################################################################################
    #   DOWNLOAD THE DATA AND SAVE IT AS A DATAFRAME - MAYBE EXCHANGE WITH SCRAPING LATER?
    ####################################################################################################################
    #Debugging method, FOR TESTING ONLY!!!:
    df_shuffled_data, df_download = functions_formatting.import_data_debugging(df_shuffled_data)

    #Live method:
    # df_download = functions_sql.select_data()


    N_new = df_download["votes"].sum()

    #If there is new data, update the sheet and the prediction
    if N_new!=N_old:
        ####################################################################################################################
        #   UPDATE DATAFRAMES OF ALL DIFFERENT GRANULARITIES
        ####################################################################################################################
        # Okrsok
        df_temp = sqldf('select town_code, okrsok, sum(votes) as votes from df_download group by town_code, okrsok')
        df_ucast_okrsok = sqldf('select a.*, b.votes from df_ucast_okrsok_historical a left join df_temp b on a.town_code = b.town_code and a.okrsok = b.okrsok')
        df_ucast_okrsok.loc[df_ucast_okrsok['votes'].isna(), 'votes'] = 0

        # Town
        df_temp = sqldf('select town_code, sum(votes) as votes, count(distinct okrsok) as count_okrsok from df_download group by town_code')
        df_ucast_town = sqldf('select a.*, b.votes, b.count_okrsok from df_ucast_town_historical a left join df_temp b on a.town_code = b.town_code')
        df_ucast_town.loc[df_ucast_town['votes'].isna(), 'votes'] = 0
        df_ucast_town.loc[df_ucast_town['count_okrsok'].isna(), 'count_okrsok'] = 0
        df_ucast_town['counted_percentage'] = df_ucast_town['count_okrsok'] / df_ucast_town['total_count_okrsok']

        # Okres
        df_temp = sqldf('select okres_code, sum(votes) as votes, count(distinct town_code || okrsok) as count_okrsok from df_download group by okres_code')
        df_ucast_okres = sqldf('select a.*, b.votes, b.count_okrsok from df_ucast_okres_historical a left join df_temp b on a.okres_code = b.okres_code')
        df_ucast_okres.loc[df_ucast_okres['votes'].isna(), 'votes'] = 0
        df_ucast_okres.loc[df_ucast_okres['count_okrsok'].isna(), 'count_okrsok'] = 0
        df_ucast_okres['counted_percentage'] = df_ucast_okres['count_okrsok'] / df_ucast_okres['total_count_okrsok']

        # Obvod
        df_temp = sqldf('select obvod_code, sum(votes) as votes, count(distinct okres_code || town_code || okrsok) as count_okrsok from df_download group by obvod_code')
        df_ucast_obvod = sqldf('select a.*, b.votes, b.count_okrsok from df_ucast_obvod_historical a left join df_temp b on a.obvod_code = b.obvod_code')
        df_ucast_obvod.loc[df_ucast_obvod['votes'].isna(), 'votes'] = 0
        df_ucast_obvod.loc[df_ucast_obvod['count_okrsok'].isna(), 'count_okrsok'] = 0
        df_ucast_obvod['counted_percentage'] = df_ucast_obvod['count_okrsok'] / df_ucast_obvod['total_count_okrsok']

        # Kraj
        df_temp = sqldf('select kraj_code, sum(votes) as votes, count(distinct obvod_code || okres_code || town_code || okrsok) as count_okrsok from df_download group by kraj_code')
        df_ucast_kraj = sqldf('select a.*, b.votes, b.count_okrsok from df_ucast_kraj_historical a left join df_temp b on a.kraj_code = b.kraj_code')
        df_ucast_kraj.loc[df_ucast_kraj['votes'].isna(), 'votes'] = 0
        df_ucast_kraj.loc[df_ucast_kraj['count_okrsok'].isna(), 'count_okrsok'] = 0
        df_ucast_kraj['counted_percentage'] = df_ucast_kraj['count_okrsok'] / df_ucast_kraj['total_count_okrsok']

        # Total
        counted_percentage = round(sqldf('select count(okrsok) from df_ucast_okrsok where votes<>0').values[0][0] / len(df_ucast_okrsok), 8)

        df_ucast_okrsok = sqldf("select a.*, b.last_year_votes as last_year_votes_town from df_ucast_okrsok a left join df_ucast_town b on a.town_code = b.town_code")
        df_ucast_okrsok["percentage_of_town"] = df_ucast_okrsok['last_year_votes'] / df_ucast_okrsok['last_year_votes_town']

        df_temp = sqldf('select town_code, sum(percentage_of_town) as counted_percentage_of_votes from df_ucast_okrsok where votes!=0 group by town_code')
        df_ucast_town = sqldf('select a.*, b.counted_percentage_of_votes from df_ucast_town a left join df_temp b on a.town_code=b.town_code')
        df_ucast_town.loc[df_ucast_town['counted_percentage_of_votes'].isna(), 'counted_percentage_of_votes'] = 0

        # Save the attendance datasets as csv files:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Saving attendance values into csv files...")
        df_ucast_okrsok.to_csv(config.data_path + "gui/df_ucast_okrsok.csv", header=True, index=False)
        df_ucast_town.to_csv(config.data_path + "gui/df_ucast_town.csv", header=True, index=False)
        df_ucast_okres.to_csv(config.data_path + "gui/df_ucast_okres.csv", header=True, index=False)
        df_ucast_obvod.to_csv(config.data_path + "gui/df_ucast_obvod.csv", header=True, index=False)
        df_ucast_kraj.to_csv(config.data_path + "gui/df_ucast_kraj.csv", header=True, index=False)

        ####################################################################################################################
        #   DATA UPDATE TIMESTAMP
        ####################################################################################################################
        data_update_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df_data_update_timestamp = pd.DataFrame([data_update_timestamp])
        df_data_update_timestamp.columns = [["timestamp"]]

        # Save the data update timestamps as csv files:
        df_data_update_timestamp.to_csv(config.data_path + "gui/df_data_update_timestamp.csv", header=True, index=False)

        ####################################################################################################################
        #   COMPUTE THE EXPECTED VOTER ATTENDANCE
        ####################################################################################################################
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Updating expected attendance...")

        #Set the variables used in computation
        df_ucast_computation = df_ucast_town[['kraj_code', 'kraj', 'obvod_code', 'obvod', 'okres_code', 'okres', 'town_code', 'last_year_votes', 'votes', 'counted_percentage']]
        df_ucast_computation.loc[df_ucast_computation['counted_percentage'].isna(), 'votes'] = 0
        df_ucast_computation.loc[df_ucast_computation['counted_percentage'].isna(), 'counted_percentage'] = config.eps

        v_hat = df_ucast_computation['last_year_votes'].values
        v_bar = df_ucast_computation['votes'].values
        c_bar = df_ucast_computation['counted_percentage'].values
        N = np.shape(A)[0]


        # Set the tilde variables, see documentation
        v_tilde = v_bar / v_hat
        A_tilde = (np.diag((1-c_bar)/A.dot(np.ones(N)))).dot(A)

        # Set the theta:
        temp_expected_ucast = np.sum(np.sign(c_bar) * v_tilde / (c_bar + config.eps)) / np.sum(np.sign(c_bar))
        lower_bar = 0.95 * (1 - counted_percentage) + 0.55 * counted_percentage
        upper_bar = 1.2 * (1 - counted_percentage) + 1.05 * counted_percentage
        if temp_expected_ucast<lower_bar:
            theta = 0.95
        else:
            if temp_expected_ucast<1:
                theta = min((temp_expected_ucast - lower_bar)/(1-lower_bar) * 0.95, 0.95)
            else:
                if temp_expected_ucast<upper_bar:
                    theta = min((upper_bar - temp_expected_ucast) / (upper_bar - 1), 0.95)
                else:
                    theta = 0.95

        # Set the final variables that go into the linear solver
        AA = scipy.sparse.csr_matrix(np.eye(N) - theta * A_tilde)
        bb = v_tilde + (1 - theta) * (1 - c_bar)

        # Solve the system and compute the expected number of votes per town
        phi = scipy.sparse.linalg.spsolve(AA, bb)
        v = phi * v_hat

        df_v = pd.DataFrame(v, columns=['expected_votes'])
        df_ucast_computation = pd.concat([df_ucast_computation,df_v], axis=1)

        ####################################################################################################################
        #  COMPUTE THE EXPECTED NUMBER AND THE EXPECTED ATTENDANCE PER TOWN, KRAJ
        ####################################################################################################################
        df_expected_ucast_town = df_ucast_computation[["town_code", "expected_votes"]]
        df_temp = pd.read_csv(config.data_path + "potential_voters_per_town.csv")
        df_expected_ucast_town = sqldf('select a.*, b.number_of_potential_voters, a.expected_votes/b.number_of_potential_voters as expected_attendance from df_expected_ucast_town a left join df_temp b on a.town_code=b.town_code')

        df_temp = pd.read_csv(config.data_path + "potential_voters_per_kraj.csv")
        df_expected_ucast_kraj = sqldf('select kraj_code, kraj, sum(expected_votes) as expected_votes from df_ucast_computation group by kraj_code')
        df_expected_ucast_kraj = sqldf('select a.*, b.number_of_potential_voters, a.expected_votes/b.number_of_potential_voters as expected_attendance from df_expected_ucast_kraj a left join df_temp b on a.kraj_code=b.kraj_code')

        ####################################################################################################################
        #  COMPUTE THE EXPECTED VOTES PER PARTY
        ####################################################################################################################
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Computing expected votes per party...")
        # Town dataset containing the party_ID and votes
        df_parties_town = sqldf('select town_code, party_ID, sum(votes) as votes from df_download group by town_code, party_ID')
        # y_star variables will be stacked here:
        y_star_set = np.array([])
        current_votes_per_party = []

        test =np.tile(v, (N, 1))
        test2 = (np.multiply(A, np.tile(v, (N, 1))))
        A_tilde = (np.diag((1 - c_bar) / A.dot(v))).dot(np.multiply(A, np.tile(v, (N, 1))))
        AA = scipy.sparse.csr_matrix(np.eye(N) - theta * A_tilde)

        party_ID_index = 10
        for party_ID_index in range(0,K):
            party_ID = df_parties['party_ID'].iloc[party_ID_index]
            #Only select one party, make sure indexing is correct by using left join to base table df_ucast_town_historical
            df_temp = sqldf('select * from df_parties_town where party_ID=' + str(party_ID))
            df_party_town = sqldf('select a.*, b.votes from df_ucast_town_historical a left join df_temp b on a.town_code = b.town_code')

            df_party_computation = df_party_town[['town_code', 'votes']]
            df_party_computation.loc[df_party_computation['votes'].isna(), 'votes'] = 0

            y_bar = df_party_computation['votes'].values / (v_bar+config.eps)

            # If y_bar is zero, so if this party has absolutely no votes at all, automatically put y_star = 0 as well.
            # Otherwise we get an error in the sparse linear solver for some reason.
            if sum(y_bar)==0:
                y_star = y_bar
            else:
                bb = c_bar * y_bar + (1 - theta) * (1 - c_bar) * y_bar

                # Solve the system and compute the expected number of votes per town
                y_star = scipy.sparse.linalg.spsolve(AA, bb)

            current_votes = sum(np.transpose(y_bar * v_bar))

            #Stack the vectors:
            y_star_set = np.vstack([y_star_set, y_star]) if y_star_set.size else y_star
            current_votes_per_party.append(current_votes)

        sum_y_star_set = sum(y_star_set)
        sum_y_star_set[sum_y_star_set == 0] = 1

        y = y_star_set / (np.tile(sum_y_star_set,(K,1)))

        expected_votes_per_party = sum(np.transpose(y * (np.tile(v,(K,1)))))

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

        ####################################################################################################################
        #  COMPUTE THE QUANTILES
        ####################################################################################################################
        quantile_floor = max(np.floor(100 * counted_percentage), 1) / 100
        quantile_ceil = np.ceil(100 * max(counted_percentage, 0.01)) / 100
        lower_part = (max(counted_percentage, 0.01) - quantile_floor) / 0.01
        upper_part = (quantile_ceil - max(counted_percentage, 0.01)) / 0.01
        if lower_part * upper_part==0:
            lower_part = 0.5
            upper_part = 0.5

        lower_quantile = upper_part * float(df_quantiles[df_quantiles["counted_percentage"] == quantile_floor]['lower_quantile'].iloc[0]) + lower_part * float(df_quantiles[df_quantiles["counted_percentage"] == quantile_ceil]['lower_quantile'].iloc[0])
        upper_quantile = upper_part * float(df_quantiles[df_quantiles["counted_percentage"] == quantile_floor]['upper_quantile'].iloc[0]) + lower_part * float(df_quantiles[df_quantiles["counted_percentage"] == quantile_ceil]['upper_quantile'].iloc[0])
        mu = upper_part * float(df_quantiles[df_quantiles["counted_percentage"] == quantile_floor]['mu'].iloc[0]) + lower_part * float(df_quantiles[df_quantiles["counted_percentage"] == quantile_ceil]['mu'].iloc[0])
        sigma_squared = upper_part * float(df_quantiles[df_quantiles["counted_percentage"] == quantile_floor]['sigma_squared'].iloc[0]) + lower_part * float(df_quantiles[df_quantiles["counted_percentage"] == quantile_ceil]['sigma_squared'].iloc[0])

        lower_prediction_quantile = expected_results / (1 - lower_quantile)
        temp = list(lower_prediction_quantile)
        temp.insert(0, counted_percentage)
        temp.insert(0, prediction_update_timestamp)
        df_lower_prediction_quantile = pd.DataFrame(temp)

        upper_prediction_quantile = expected_results / (1 - upper_quantile)
        temp = list(upper_prediction_quantile)
        temp.insert(0, counted_percentage)
        temp.insert(0, prediction_update_timestamp)
        df_upper_prediction_quantile = pd.DataFrame(temp)

        ####################################################################################################################
        #  COMPUTE THE PROBABILITY OF WINNING
        ####################################################################################################################
        # df_probability_computations = pd.DataFrame([counted_percentage, lower_quantile, upper_quantile, mu, sigma_squared]).transpose()
        mu_y = expected_results
        sigma_y = expected_results * np.sqrt(sigma_squared)
        mu_combination = np.tile(mu_y,[K,1]).transpose() - np.tile(mu_y,[K,1])
        sigma_combination = np.tile(sigma_y, [K, 1]).transpose() + np.tile(sigma_y, [K, 1])
        sigma_combination = sigma_combination + (1 - np.sign(sigma_combination))*config.eps

        # Probability that the k1-th party has more votes than the k2-th party:
        P = 0*mu_combination
        for k1 in range(0,K):
            for k2 in range(0, K):
                P[k1][k2] = max(1 - norm.cdf(-mu_combination[k1][k2]/sigma_combination[k1,k2]),config.eps)
        df_P = pd.DataFrame(P)

        # Fair betting odds:
        odds = 1/P
        df_odds = pd.DataFrame(odds)

        # Probability that the k-th party has more votes than everyone else. THIS IS NOT CORRECT AS THESE ARE NOT INDEPENDENT BUT IT IS FUCKING FAST AS SHIT:
        P_win = np.prod(P, axis=1)
        P_win = P_win / np.sum(P_win)
        temp = list(P_win)
        temp.insert(0, counted_percentage)
        temp.insert(0, prediction_update_timestamp)
        df_P_win = pd.DataFrame(temp)

        ####################################################################################################################
        #  SAVE THE RELEVANT VARIABLES AS CSV FILES
        ####################################################################################################################
        # Save the prediction as csv files:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Appending / rewriting the GUI csv files..")

        df_expected_ucast_town[["expected_attendance"]].transpose().to_csv(config.data_path + "gui/df_expected_ucast_town.csv", header=False, index=False, mode='a')
        df_expected_ucast_kraj[["expected_attendance"]].transpose().to_csv(config.data_path + "gui/df_expected_ucast_kraj.csv", header=False, index=False, mode='a')

        df_expected_results.transpose().to_csv(config.data_path + "gui/df_expected_results.csv", header=False, index=False, mode='a')
        df_current_results.transpose().to_csv(config.data_path + "gui/df_current_results.csv", header=False, index=False, mode='a')
        df_lower_prediction_quantile.transpose().to_csv(config.data_path + "gui/df_lower_prediction_quantile.csv", header=False, index=False, mode='a')
        df_upper_prediction_quantile.transpose().to_csv(config.data_path + "gui/df_upper_prediction_quantile.csv", header=False, index=False, mode='a')

        df_prediction_update_timestamp = pd.DataFrame([data_update_timestamp])
        df_prediction_update_timestamp.columns = [["timestamp"]]

        # Save the data update timestamps as csv files:
        df_prediction_update_timestamp.to_csv(config.data_path + "gui/df_prediction_update_timestamp.csv", header=True, index=False)

        # Save the probabilistic variables as csv files:
        df_P_win.transpose().to_csv(config.data_path + "gui/df_P_win.csv", header=False, index=False, mode='a')
        df_P.transpose().to_csv(config.data_path + "gui/df_P.csv", header=False, index=False)
        df_odds.transpose().to_csv(config.data_path + "gui/df_odds.csv", header=False, index=False)

        ####################################################################################################################
        #  FINISH THE CYCLE
        ####################################################################################################################
        if df_ucast_town['counted_percentage'].mean()==1:
            finished = 1
        N_old = N_new
        print(datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + ": " + "Cycle finished. Currently accounted for " + str(100*counted_percentage) + " % of voting places.")
    else:
        print(datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + ": " + "Waiting for new data...")
        time.sleep(3)

