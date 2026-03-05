import os
import requests
import msal
from config import (
    MS_CLIENT_ID, MS_CLIENT_SECRET, MS_TENANT_ID, 
    MS_DRIVE_ID, MS_FOLDER_ID, INPUT_DIR
)

def get_access_token():
    """Authenticate with Azure App Registration to get an access token."""
    authority = f"https://login.microsoftonline.com/{MS_TENANT_ID}"
    app = msal.ConfidentialClientApplication(
        MS_CLIENT_ID,
        authority=authority,
        client_credential=MS_CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"Failed to acquire token: {result.get('error_description')}")

def sync_drive_files():
    """
    Downloads new .pdf and .xlsx files from the specified Microsoft Drive folder 
    directly into the local data/input folder.
    """
    print("Authenticating with Microsoft Graph API...")
    try:
        token = get_access_token()
    except Exception as e:
        print(f"Skipping drive sync (credentials probably not set): {e}")
        return

    headers = {"Authorization": f"Bearer {token}"}
    
    # Endpoint to list children of a folder
    url = f"https://graph.microsoft.com/v1.0/drives/{MS_DRIVE_ID}/items/{MS_FOLDER_ID}/children"
    
    print("Fetching files from Drive...")
    # Include a network timeout to avoid hanging indefinitely.
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        print(f"Error fetching files: {response.text}")
        return
        
    items = response.json().get('value', [])
    
    for item in items:
        name = item.get('name', '')
        if name.endswith('.pdf') or name.endswith('.xlsx'):
            download_url = item.get('@microsoft.graph.downloadUrl')
            if not download_url:
                continue
                
            local_path = INPUT_DIR / name
            # Skip if already downloaded (basic check)
            if local_path.exists():
                print(f"File {name} already exists. Skipping.")
                continue
                
            print(f"Downloading {name}...")
            # Include a network timeout to avoid hanging indefinitely.
            file_response = requests.get(download_url, timeout=30)
            with open(local_path, "wb") as f:
                f.write(file_response.content)
                
    print(f"Sync complete. Files saved to {INPUT_DIR}")

if __name__ == "__main__":
    sync_drive_files()
