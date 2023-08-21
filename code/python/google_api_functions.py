import sys
sys.path.append("//media/juro/DATA/Work/Elections/code/python")

from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os
import pandas as pd

import config

def establish_connection():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(config.data_path +
                                                             'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('sheets', 'v4', credentials=creds)
    service.spreadsheets()
    return service

def upload_dataframe(df, range, spreadsheet_id):
    values = df.values.tolist()

    # Call the Sheets API
    body = {
        'values': values
    }

    service = establish_connection()

    result = service.spreadsheets().values().update(spreadsheetId=spreadsheet_id,
                                                    range=range, valueInputOption="USER_ENTERED",
                                                    body=body).execute()
    print(('{} cells updated.'.format(result.get('updatedCells'))))

def append_dataframe(df, range, spreadsheet_id):
    values = df.values.tolist()

    # Call the Sheets API
    body = {
        'values': values
    }

    service = establish_connection()

    result = service.spreadsheets().values().append(spreadsheetId=spreadsheet_id,
                                                    range=range, valueInputOption="USER_ENTERED",
                                                    body=body).execute()

def get_dataframe(range, spreadsheet_id):
    service = establish_connection()

    # # Call the Sheets API
    result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id,
                                                 range=range).execute()
    values = result.get('values', [])

    df = pd.DataFrame(values)
    return df

def clear_range(range, spreadsheet_id):
    service = establish_connection()

    # # Call the Sheets API
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + "Clearing range " + range)
    service.spreadsheets().values().clear(spreadsheetId=spreadsheet_id,
                                                 range=range).execute()

def clear_sheet(spreadsheet_id):
    clear_range(config.RANGE_CURRENT_RESULTS, spreadsheet_id)
    clear_range(config.RANGE_GRANULAR_DATA_WRITE, spreadsheet_id)
    clear_range(config.RANGE_PREDICTION, spreadsheet_id)
    clear_range(config.RANGE_LOWER_QUANTILE, spreadsheet_id)
    clear_range(config.RANGE_UPPER_QUANTILE, spreadsheet_id)

    # clear_range(config.RANGE_UCAST_OKRSOK)
    # clear_range(config.RANGE_UCAST_TOWN)
    # clear_range(config.RANGE_UCAST_OKRES)
    # clear_range(config.RANGE_UCAST_OBVOD)
    # clear_range(config.RANGE_UCAST_KRAJ)

    # clear_range(config.RANGE_DASHBOARD_CURRENT_RESULTS)
    # clear_range(config.RANGE_DASHBOARD_CURRENT_PREDICTION)
    # clear_range(config.RANGE_DASHBOARD_LAST_PREDICTION)
    # clear_range(config.RANGE_DASHBOARD_LAST_PREDICTION_UPDATE)
    # clear_range(config.RANGE_DASHBOARD_LAST_DATA_UPDATE)

def clear_lite_sheet(spreadsheet_id):
    clear_range(config.RANGE_CURRENT_RESULTS, spreadsheet_id)
    clear_range(config.RANGE_PREDICTION, spreadsheet_id)
    clear_range(config.RANGE_LOWER_QUANTILE, spreadsheet_id)
    clear_range(config.RANGE_UPPER_QUANTILE, spreadsheet_id)

