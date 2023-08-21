from datetime import datetime
import mysql
import mysql.connector
import pandas as pd
import subprocess
from subprocess import call

import config

####################################################################################################################
#   TECHNICAL SQL FUNCTIONS
####################################################################################################################

def sql_alter_encoding():
    db = mysql.connector.connect(host='localhost',
                                             database='elections',
                                             user='elector',
                                             password='frufru',
                                             auth_plugin='mysql_native_password')
    cursor = db.cursor()
    cursor.execute("ALTER DATABASE `%s` CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_unicode_ci'" % "flat_finder_db")


    sql = "SET NAMES utf8mb4;"
    cursor.execute(sql)
    sql = "SELECT DISTINCT(table_name) FROM information_schema.columns WHERE table_schema = 'flat_fnder_db'"
    cursor.execute(sql)

    results = cursor.fetchall()
    for row in results:
        sql = "ALTER TABLE `%s` convert to character set DEFAULT COLLATE DEFAULT" % (row[0])
        cursor.execute(sql)
    db.close()

def sql_connect(sql_query, sql_tuple, data_extracted, success_message):
    result = 0
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='elections',
                                             user='elector',
                                             password='frufru',
                                             auth_plugin='mysql_native_password')
        cursor = connection.cursor(buffered=True)
        cursor.execute("SET NAMES utf8mb4;")  # or utf8 or any other charset you want to handle
        cursor.execute("SET CHARACTER SET utf8mb4;")  # same as above
        cursor.execute("SET character_set_connection=utf8mb4;")  # same as above
        cursor.execute(sql_query, sql_tuple)
        connection.commit()
        if data_extracted==True:
            result = cursor.fetchall()
        else:
            result = None
        if success_message:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + success_message)

    except mysql.connector.Error as error :
        # rollback if any exception occurred
        connection.rollback()
        print("Query failed at table {}".format(error))

    finally:
        # closing database connection.
        if connection.is_connected():
            cursor.close()
            connection.close()
            # print("MySQL connection is closed")

    return result

####################################################################################################################
#   TABLE CREATION
####################################################################################################################
def create_table_current_data():
    data_extracted = False
    sql_query = "create table current_data (timestamp text, finished_percentage text, kraj_code int, obvod_code int, okres_code int, town_code int, okrsok int, party_ID int, party_name text, votes numeric, votes_percentage text, preferential_votes text, preferential_votes_percentage text) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
    sql_connect(sql_query, None, data_extracted, success_message = "Table current_data created.")
#
# def create_table_last_update():
#     data_extracted = False
#     sql_query = "create table last_update (last_data_update_timestamp text, last_prediction_update_timestamp text) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
#     sql_connect(sql_query, None, data_extracted, success_message = "Table last_update created.")
#
# def create_table_predictions(list_parties):
#     data_extracted = False
#     sql_query = "CREATE TABLE predictions (timestamp text, counted_percentage text, "
#     for party_index in range(0, len(list_parties)):
#         party = list_parties[party_index]
#         sql_query = sql_query + party + " text, "
#     sql_query = sql_query[:-2]
#     sql_query = sql_query + ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
#     sql_connect(sql_query, None, data_extracted, success_message="Table predictions created.")

####################################################################################################################
#   TABLE DROPS
####################################################################################################################
def drop_table(table_name):
    data_extracted = False
    sql_query = "DROP TABLE " + table_name + ";"
    sql_connect(sql_query, None, data_extracted, success_message = "Table " + table_name + " dropped.")

####################################################################################################################
#   INSERT DATA
####################################################################################################################
def insert_data(timestamp, finished_percentage, kraj_code, obvod_code, okres_code, town_code, okrsok, party_ID, party_name, votes, votes_percentage, preferential_votes, preferential_votes_percentage):
    data_extracted = False
    sql_query = """ INSERT INTO current_data
                           (timestamp, finished_percentage, kraj_code, obvod_code, okres_code, town_code, okrsok, party_ID, party_name, votes, votes_percentage, preferential_votes, preferential_votes_percentage) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    sql_tuple = (timestamp, finished_percentage, kraj_code, obvod_code, okres_code, town_code, okrsok, party_ID, party_name, votes, votes_percentage, preferential_votes, preferential_votes_percentage)
    sql_connect(sql_query, sql_tuple, data_extracted, success_message = False)
#
# def insert_last_update(last_data_update_timestamp, last_prediction_update_timestamp):
#     data_extracted = False
#     sql_query = """ INSERT INTO last_update (last_data_update_timestamp, last_prediction_update_timestamp) VALUES (%s,%s)"""
#     sql_tuple = (last_data_update_timestamp, last_prediction_update_timestamp)
#     sql_connect(sql_query, sql_tuple, data_extracted, success_message = False)
#
# def insert_prediction(new_row_tuple, list_parties):
#     data_extracted = False
#     sql_query = "INSERT INTO predictions (timestamp, counted_percentage, "
#     for party_index in range(0, len(list_parties)):
#         party = list_parties[party_index]
#         sql_query = sql_query + party + ", "
#     sql_query = sql_query[:-2]
#     sql_query = sql_query + ") VALUES (%s, %s, "
#     for party_index in range(0, len(list_parties)):
#         sql_query = sql_query + "%s, "
#     sql_query = sql_query[:-2]
#     sql_query = sql_query + ");"
#     sql_tuple = new_row_tuple
#     sql_connect(sql_query, sql_tuple, data_extracted, success_message="Table predictions created.")

####################################################################################################################
#   CLEAR DATA
####################################################################################################################
def delete_all_data():
    data_extracted = False
    sql_query = "DELETE FROM current_data;"
    sql_connect(sql_query, None, data_extracted, success_message="All data deleted.")

# def delete_last_update():
#     data_extracted = False
#     sql_query = "DELETE FROM last_update;"
#     sql_connect(sql_query, None, data_extracted, success_message="Last update timestamps deleted.")

####################################################################################################################
#   SELECT DATA
####################################################################################################################
def select_data():
    data_extracted = True
    sql_query = " SELECT * FROM current_data;"
    list_result = sql_connect(sql_query, None, data_extracted, success_message="Data loaded from the database.")
    df_result = pd.DataFrame(list_result)
    df_result.columns = ["timestamp",
                        "counted_percentage",
                        "kraj_code",
                        "obvod_code",
                        "okres_code",
                        "town_code",
                        "okrsok",
                        "party_ID",
                        "party_name",
                        "votes",
                        "votes_percentage",
                        "preferential_votes",
                        "preferential_votes_percentage"]
    return df_result

def select_last_update():
    data_extracted = True
    sql_query = " SELECT * FROM last_update;"
    result = sql_connect(sql_query, None, data_extracted, success_message="Last update timestamps loaded from the database.")
    return result
#
# def select_predictions():
#     data_extracted = True
#     sql_query = " SELECT * FROM predictions;"
#     result = sql_connect(sql_query, None, data_extracted, success_message="All predictions loaded from the database.")
#     return result
#
# def select_last_prediction():
#     data_extracted = True
#     sql_query = " SELECT * FROM predictions a WHERE a.timestamp IN (SELECT last_prediction_update_timestamp FROM last_update b);"
#     result = sql_connect(sql_query, None, data_extracted, success_message="All predictions loaded from the database.")
#     return result






















