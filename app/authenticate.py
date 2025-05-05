import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    creds = None
    token_file = 'token.json'
    
    # Check if token exists and attempt to load it
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as token:
                creds = Credentials.from_authorized_user_info(json.load(token), SCOPES)
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
        except Exception as e:
            print(f"Error loading token.json: {e}")
            os.remove(token_file)
    
    # If no valid creds, start the flow
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', 
            SCOPES,
            redirect_uri='http://localhost:8502'
        )
        creds = flow.run_local_server(
            port=8502,
            access_type='offline',
            prompt='consent'
        )
        
        # Save the credentials
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
        print("Authentication successful. token.json has been created with refresh_token.")
    
    return creds

if __name__ == "__main__":
    authenticate()