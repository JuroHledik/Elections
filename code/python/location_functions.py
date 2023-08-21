import sys
sys.path.append("//media/juro/DATA/Work/Elections/code/python")

from datetime import datetime
from math import sin, cos, sqrt, asin, radians
import numpy as np
import pandas as pd
from pandasql import sqldf

import config

def GPS_distance(lat1, lon1 , lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    distance = R * c
    return distance

def get_neighbor_matrix(df_temp):
    #Import the town data:
    df_town_data = pd.read_csv(config.data_path + "towns.csv", delimiter=';')

    #Save the IDs and store them:
    df_towns = sqldf('select a.*,b.latitude,b.longitude from df_temp a left join df_town_data b on a.town_code = b.obec_kod')

    N = len(df_towns)

    # First, we create the distance matrix for the whole city if it does not exist already:
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "+ "Creating the distance matrix...")
    LatList = df_towns['latitude'].tolist()
    LonList = df_towns['longitude'].tolist()
    DistanceMatrix = config.MaxPossibleDistance * np.eye(N)
    for n1 in range(0, N):
        # print(n1)
        for n2 in range(0, n1 + 1):
            dist = GPS_distance(LatList[n1], LonList[n1], LatList[n2], LonList[n2])
            if n1 != n2:
                DistanceMatrix[n1, n2] = dist
                DistanceMatrix[n2, n1] = dist
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "+ "Distance matrix created.")



    # Next, we create the neighbor matrix for the main algorithm (basically binary adjacency matrix):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "+ "Creating the neighbor matrix...")
    NeighborMatrix = np.zeros((N, N))
    max_dist = max([max(l) for l in DistanceMatrix])
    TempMatrix = np.copy(DistanceMatrix)
    TempMatrix = np.ma.array(TempMatrix, mask=np.isnan(TempMatrix)) #This disregards NANs in the next lines.
    n=0
    for n in range(0, N):
        IterationFinished = False
        while IterationFinished == False:
            CurrentRow = TempMatrix[n, :]
            LowestDistanceIndex = np.argmin(CurrentRow)
            LowestDistance = TempMatrix[n,LowestDistanceIndex]

            if LowestDistance < max_dist and np.sum(NeighborMatrix[n, :]) < config.MaxNumberOfNeighbors:
                NeighborMatrix[n, LowestDistanceIndex] = 1
                TempMatrix[n, LowestDistanceIndex] = max_dist
            else:
                IterationFinished = True
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "+ "Neighbor matrix created.")

    return NeighborMatrix