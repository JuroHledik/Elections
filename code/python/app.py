#!/usr/bin/env python3
from __future__ import print_function

import sys
# sys.path.append("//media/juro/DATA/Work/Elections/code/python")
sys.path.append("//mnt/b37936f4-84ca-4659-b59a-3142ded0de7c/Projects/Elections/code/python")

from dash import Dash, dcc, html, Input, Output
from distinctipy import distinctipy

import pandas as pd
import plotly
import random
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import deque

import config

####################################################################################################################
#   FUNCTIONS
####################################################################################################################
def import_data():
    df_parties = pd.read_csv(config.data_path + "parties.csv")
    df_parties.columns = ['party_ID', 'party_name']

    df_current_results = pd.read_csv(config.data_path + "gui/df_current_results.csv", header=None)
    temp_columns = ['timestamp', 'counted_percentage']
    temp_columns.extend(df_parties["party_name"].to_list())
    df_current_results.columns = temp_columns
    df_current_results_only_parties = df_current_results.drop(['timestamp', 'counted_percentage'], axis=1)

    last_results = df_current_results.iloc[-1].to_list()
    last_results.pop(0)
    last_results.pop(0)

    previous_results = df_current_results.iloc[-2].to_list()
    previous_results.pop(0)
    previous_results.pop(0)

    counted_percentage = df_current_results[["counted_percentage"]].iloc[-1][0]
    timestamp = df_current_results[["timestamp"]].iloc[-1][0]

    df_expected_results = pd.read_csv(config.data_path + "gui/df_expected_results.csv", header=None)
    temp_columns = ['timestamp', 'counted_percentage']
    temp_columns.extend(df_parties["party_name"].to_list())
    df_expected_results.columns = temp_columns
    df_expected_results_only_parties = df_expected_results.drop(['timestamp', 'counted_percentage'], axis=1)

    last_results = df_expected_results.iloc[-1].to_list()
    last_results.pop(0)
    last_results.pop(0)

    previous_results = df_expected_results.iloc[-2].to_list()
    previous_results.pop(0)
    previous_results.pop(0)

    dict_data = {'counted_percentage': counted_percentage,
                 'df_current_results': df_current_results,
                 'df_current_results_only_parties': df_current_results_only_parties,
                 'df_expected_results': df_expected_results,
                 'df_expected_results_only_parties': df_expected_results_only_parties,
                 'df_parties': df_parties,
                 'last_results': last_results,
                 'previous_results': previous_results,
                 'timestamp': timestamp}

    return dict_data


####################################################################################################################
#   IMPORT DATA AN INITIATE GRAPHS
####################################################################################################################
app = Dash(__name__)

dict_data = import_data()
list_parties = dict_data["df_parties"]["party_name"].to_list()

app.layout = html.Div([
    html.Div(id='text_current_state'),
    html.H4('Vyberte strany ktore chcete sledovat:'),
    dcc.Dropdown(
        id="parties_dropdown",
        options=list_parties,
        value=list_parties,
        multi=True
    ),
    html.Div([
        html.Div([
            dcc.Graph(id="graph_current_results", animate=False),
        ], style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id="graph_expected_results", animate=False)
        ], style={'width': '49%', 'display': 'inline-block'})
    ], className="row"),

    # html.Div(children=[
    #     dcc.Graph(id="graph_current_results", style={'display': 'inline-block'}, animate=False),
    #     dcc.Graph(id="graph_expected_results", style={'display': 'inline-block'}, animate=False)
    #     ]
    # ),
    dcc.Interval(
            id = 'update',
            interval = 1000,
            n_intervals = 0
        )
])


@app.callback(Output('text_current_state', 'children'),
              Input('update', 'n_intervals'))

def update_state_current_state(n):
    dict_data = import_data()
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Posledná aktualizácia: ' + dict_data["timestamp"], style=style),
        html.Span('Zrátaných okrskov: {0:.4f}'.format(100 *dict_data["counted_percentage"]) + ' %', style=style),
    ]

@app.callback(
    Output("graph_current_results", "figure"),
    Input("update", "n_intervals"),
    Input("parties_dropdown", "value"))

def update_graph_current_results(n_intervals, list_selected_parties):
    dict_data = import_data()
    relevant_columns = ["timestamp", "counted_percentage"]
    relevant_columns.extend(list_selected_parties)
    df_current_results = dict_data["df_current_results"][relevant_columns]
    last_results = df_current_results.iloc[-1].to_list()
    last_results.pop(0)
    last_results.pop(0)
    colors = distinctipy.get_colors(len(list_selected_parties), rng=config.random_seed)

    last_results, list_selected_parties, colors = zip(*sorted(zip(last_results, list_selected_parties, colors)))
    list_selected_parties = list(list_selected_parties)
    list_selected_parties.reverse()

    fig = make_subplots(rows=1, cols=1)

    for party_index in range(0,len(list_selected_parties)):
        fig.add_scatter(x=df_current_results['counted_percentage'],
                        y=df_current_results[list_selected_parties[party_index]],
                        row=1,
                        col=1,
                        line=dict(color='rgb(' + str(round(255*colors[party_index][0])) + ',' + str(round(255*colors[party_index][1])) + ',' + str(round(255*colors[party_index][2])) + ')'))

    fig.update_layout(
        title="Aktuálne výsledky",
        xaxis_title="Pomer zrátaných okrskov",
        yaxis_title="Vývoj",
        legend_title="Strany",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="Black"
        )
    )
    for index_party in range(0,len(list_selected_parties)):
        fig.data[index_party].name = list_selected_parties[index_party]
        fig.data[index_party].hovertemplate = list_selected_parties[index_party]
    return fig

@app.callback(
    Output("graph_expected_results", "figure"),
    Input("update", "n_intervals"),
    Input("parties_dropdown", "value"))

def update_graph_expected_results(n_intervals, list_selected_parties):
    dict_data = import_data()
    relevant_columns = ["timestamp", "counted_percentage"]
    relevant_columns.extend(list_selected_parties)
    df_expected_results = dict_data["df_expected_results"][relevant_columns]
    last_results = df_expected_results.iloc[-1].to_list()
    last_results.pop(0)
    last_results.pop(0)
    colors = distinctipy.get_colors(len(list_selected_parties), rng=config.random_seed)

    last_results, list_selected_parties, colors = zip(*sorted(zip(last_results, list_selected_parties, colors)))
    list_selected_parties = list(list_selected_parties)
    list_selected_parties.reverse()

    fig = make_subplots(rows=1, cols=1)

    for party_index in range(0,len(list_selected_parties)):
        fig.add_scatter(x=df_expected_results['counted_percentage'],
                        y=df_expected_results[list_selected_parties[party_index]],
                        row=1,
                        col=1,
                        line=dict(color='rgb(' + str(round(255*colors[party_index][0])) + ',' + str(round(255*colors[party_index][1])) + ',' + str(round(255*colors[party_index][2])) + ')')
                        )

    fig.update_layout(
        title="Predikcia",
        xaxis_title="Pomer zrátaných okrskov",
        yaxis_title="Vývoj",
        legend_title="Strany",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="Black"
        )
    )
    for index_party in range(0,len(list_selected_parties)):
        fig.data[index_party].name = list_selected_parties[index_party]
        fig.data[index_party].hovertemplate = list_selected_parties[index_party]
    return fig

app.run_server(debug=True)










