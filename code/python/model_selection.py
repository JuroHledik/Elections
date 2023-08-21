#!/usr/bin/env python3
from __future__ import print_function

import sys
sys.path.append("//media/juro/DATA/Work/Elections/code/python")

import copy
from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from math import isnan
import numpy as np
from numpy import matlib
import os.path
import pandas as pd
from pandasql import sqldf
import scipy.sparse
import scipy.sparse.linalg

import config
import google_api_functions
import location_functions
import plotting_functions
import seaborn as sns

####################################################################################################################
#   DOWNLOAD THE PARTY NUMBERS AND NAMES
####################################################################################################################
service = google_api_functions.establish_connection()

# # Call the Sheets API
result = service.spreadsheets().values().get(spreadsheetId=config.SPREADSHEET_ID,
                            range=config.RANGE_PARTY_NAMES).execute()
values = result.get('values', [])

#Find the number of parties:
df_parties = pd.DataFrame(values, columns = ['party_ID', 'party_name'])

########################################################################################################################
#   SELECTING THE PROPER VALUE OF v
########################################################################################################################

# theta_set = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# low_set = [0.5,0.6,0.7,0.8,0.9,1]
# high_set = [1,1.1,1.2,1.3,1.4,1.5]
#
#
# # theta_set = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
# # low_set = [0.5,0.7,0.9]
# # high_set = [1.1,1.3,1.5]
#
# K=len(theta_set)
# df = pd.DataFrame(columns=['low', 'high', 'theta', 'phase','squared_difference'])
# phase = 0
# for phase in range(0,4):
#     for low in low_set:
#         for high in high_set:
#         # for theta in theta_set:
#         #     print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Wrangling data for low = " + str(low) + ", high = " + str(high) + ", phase = " + str(phase) + " and theta = " + str(theta) + ".")
#             print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Wrangling data for low = " + str(
#             low) + ", high = " + str(high) + ", phase = " + str(phase) + ".")
#             # df_squared_difference = pd.read_csv(config.data_path + "simulation/theta_" + str(theta) + "low_" + str(low) + "high_" + str(high) + ".csv").iloc[:,[phase]]
#             df_squared_difference = pd.read_csv(
#             config.data_path + "simulation/low_" + str(low) + "high_" + str(
#                 high) + ".csv").iloc[:, [phase]]
#             N = len(df_squared_difference)
#             df_squared_difference.columns=['squared_difference']
#             df_phase = pd.DataFrame(columns = ['phase'])
#             for i in range(0,N):
#                 df_phase.loc[-1] = [phase]
#                 df_phase.index = df_phase.index + 1  # shifting index
#                 df_phase = df_phase.sort_index()
#
#             # df_theta = pd.DataFrame(columns=['theta'])
#             # for i in range(0, N):
#             #     df_theta.loc[-1] = [theta]
#             #     df_theta.index = df_theta.index + 1  # shifting index
#             #     df_theta = df_theta.sort_index()
#
#             df_low = pd.DataFrame(columns=['low'])
#             for i in range(0, N):
#                 df_low.loc[-1] = [low]
#                 df_low.index = df_low.index + 1  # shifting index
#                 df_low = df_low.sort_index()
#
#             df_high = pd.DataFrame(columns=['high'])
#             for i in range(0, N):
#                 df_high.loc[-1] = [high]
#                 df_high.index = df_high.index + 1  # shifting index
#                 df_high = df_high.sort_index()
#
#             # df_temp = pd.concat([df_low, df_high, df_theta, df_phase, df_squared_difference], axis=1)
#             df_temp = pd.concat([df_low, df_high, df_phase, df_squared_difference], axis=1)
#             df = pd.concat([df,df_temp], axis=0)
#
# for low in low_set:
#     for high in high_set:
#         # print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Printing figure for low = " + str(
#         #     low) + ", high = " + str(high) + ", phase = " + str(phase) + " and theta = " + str(theta) + ".")
#         print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Printing figure for low = " + str(
#             low) + ", high = " + str(high) + ", phase = " + str(phase) + ".")
#         # df_temp = sqldf("select * from df where low=" + str(low) + " and high=" + str(high) + " and phase=" + str(phase) + " and theta<1.5")
#         df_temp = sqldf("select * from df where low=" + str(low) + " and high=" + str(high))
#         figures_path = config.root_path + "figures/"
#         figure_name = "low_" + str(low) + "_high_" + str(high)
#         figure_title = "low_" + str(low) + "_high_" + str(high)
#         x_variable = 'phase'
#         y_variable = 'squared_difference'
#         x_label = 'phase'
#         y_label = 'squared_difference'
#         x_variable_label = 'phase'
#         y_variable_label = 'squared_difference'
#         plotting_functions.box_plot(df_temp, figures_path, figure_name, figure_title, x_variable, x_variable_label, x_label, y_variable, y_label)

########################################################################################################################
#   SELECTING THE PROPER VALUE OF y
########################################################################################################################
columns = ['low','high','sigma','simulation','phase', 'party_ID', 'squared_difference', 'difference']
# sigma_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# low_set = [0.7, 0.9]
# high_set = [1.1, 1.3]
sigma_set = [0.5]
low_set = [0.7]
high_set = [1.3]
df = pd.DataFrame(columns=columns)
df_temp_temp = pd.DataFrame(columns=columns)
N = 100 #Number of phases
last_df_index = 0

sigma = 0.5
low = 0.7
high = 1.3
phase = 0
simulation = 0
relevant_parties = [2, 3, 5, 8, 10 ,11, 14, 16, 17, 18, 21]
for sigma in sigma_set:
    for low in low_set:
        for high in high_set:
            for simulation in range(1,150):
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Wrangling data for sigma = " + str(sigma) + ", low = " + str(low) + ", high = " + str(high) + ", simulation = " + str(simulation) + ".")
                df_temp = pd.read_csv(config.data_path + "simulation_y/y" + "_sigma_" + str(sigma) + "_low_" + str(low) + "_high_" + str(high) + "_simulation_" + str(simulation) +".csv")
                K = df_temp.shape[1]
                for phase in range(0, N):
                    for party_ID in relevant_parties:
                        # squared_difference = ((df_temp.iloc[phase,:] - df_temp.iloc[N-1,:])/df_temp.iloc[N-1,:])**2
                        squared_difference = ((df_temp.iloc[phase, party_ID] - df_temp.iloc[N - 1, party_ID]) / df_temp.iloc[N - 1,party_ID]) ** 2
                        difference = ((df_temp.iloc[phase, party_ID] - df_temp.iloc[N - 1, party_ID]) / df_temp.iloc[N - 1, party_ID])
                        # mean_squared_difference = np.max(squared_difference.iloc[relevant_parties])
                        a_row = pd.Series([low, high, sigma, simulation, phase, party_ID, squared_difference, difference])
                        row_df = pd.DataFrame([a_row])
                        row_df.columns = columns
                        df = pd.concat([df, row_df])


df.loc[df['squared_difference'].isna(), 'squared_difference'] = 0
df_backup = copy.deepcopy(df)
df = copy.deepcopy(df_backup)

df_quantile = pd.DataFrame(columns= ['low','high','sigma','phase','difference_lower_quantile','difference_upper_quantile', 'mu', 'sigma_squared'])
for sigma in sigma_set:
    for low in low_set:
        for high in high_set:
            for phase in range(0,N):
                df_temp = df[(df["low"]==low) & (df["high"]==high) & (df["sigma"]==sigma) & (df["phase"]==phase)]
                a_row = pd.Series([low, high, sigma, phase, df_temp.difference.quantile(0.05), df_temp.difference.quantile(0.95), df_temp.difference.mean(), df_temp.difference.var()])
                row_df = pd.DataFrame([a_row])
                row_df.columns = ['low','high','sigma','phase','difference_lower_quantile','difference_upper_quantile', 'mu', 'sigma_squared']
                df_quantile = pd.concat([df_quantile, row_df])


#Only select Sas, sme rodina, za ludi, sns, olano, ps_spolu atd.
# df = sqldf("select * from df where party_ID in (3,4,6,9,11,12,15,17,18,19,22)")

# party_ID = 3
for low in low_set:
    for high in high_set:
        for phase in range(0, N):
            df_temp = sqldf("select * from df where  phase=" + str(phase) + " and low=" + str(low) + " and high=" + str(high))
            df_temp2 = sqldf("select * from df_quantile where  phase=" + str(phase) + " and low=" + str(low) + " and high=" + str(high))
            figures_path = config.root_path + "figures/"
            figure_name = "low_" + str(low) + "_high_" + str(high) + "_phase_" + str(phase)
            figure_title = "low_" + str(low) + "_high_" + str(high) + "_phase_" + str(phase)
            x_variable = 'difference'
            # y_variable = ''
            x_label = 'difference'
            # y_label = 'squared_difference'
            x_variable_label = 'difference'
            # y_variable_label = 'squared_difference'
            plotting_functions.hist_plot(df_temp, figures_path, figure_name, figure_title, x_variable, x_variable_label, x_label, float(df_temp2["mu"]) , float(df_temp2["sigma_squared"]))

