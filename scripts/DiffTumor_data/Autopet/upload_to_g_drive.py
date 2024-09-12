# Step 1: Install the required libraries
# !pip install pydrive2 google-auth

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import json

# Step 2: Authenticate using the service account
def authenticate_with_service_account(json_keyfile_name):
    """
        Google Drive service with a service account.
        note: for the service account to work, you need to share the folder or
        files with the service account email.

        :return: google auth
        """
    # Define the settings dict to use a service account
    # We also can use all options available for the settings dict like
    # oauth_scope,save_credentials,etc.
    settings = {
        "client_config_backend": "service",
        "service_config": {
            "client_json_file_path": json_keyfile_name,
        }
    }
    # Create instance of GoogleAuth
    gauth = GoogleAuth()#settings=settings)
    # Authenticate
    gauth.LocalWebserverAuth()
    #gauth.ServiceAuth()
    return GoogleDrive(gauth)

# Create a folder in Google Drive
def create_drive_folder(drive, folder_name):
    # Check if folder already exists
    folder_list = drive.ListFile({'q': f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder'"}).GetList()
    if folder_list:
        # Folder already exists, return its ID
        return folder_list[0]['id']
    else:
        # Create a new folder
        folder_metadata = {
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        return folder['id']

# Load the record of uploaded files
def load_uploaded_files(record_file):
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            return json.load(f)
    return []

# Save the record of uploaded files
def save_uploaded_files(record_file, uploaded_files):
    with open(record_file, 'w') as f:
        json.dump(uploaded_files, f, indent=2)

# Step 3: Upload files to Google Drive
def upload_files_to_drive(drive, directory_path, folder_id, record_file):
    uploaded_files = load_uploaded_files(record_file)
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_path = os.path.join(directory_path, filename)
            gfile = drive.CreateFile({'title': filename, 'parents': [{'id': folder_id}]})
            gfile.SetContentFile(file_path)
            gfile.Upload()
            print(f"Uploaded {filename} to Google Drive folder ID {folder_id}")
            # Verify upload
            if 'id' in gfile:
                print(f"File ID: {gfile['id']}")
                uploaded_files.append(filename)
                save_uploaded_files(record_file, uploaded_files)
            else:
                print(f"Failed to upload {filename}")


if __name__ == '__main__':
    # Specify the path to the service account JSON key file
    json_keyfile_name = 'prepro-all-synthesized-30-6cfa14c7ad9d.json'

    # Authenticate and create a Google Drive service
    drive = authenticate_with_service_account(json_keyfile_name)

    folder_name = 'imagesTr_Step_3_synthesized'
    folder_id = create_drive_folder(drive, folder_name)

    # Specify the directory containing the files
    directory_path = 'imagesTr_Step_3_synthesized'
    record_file = 'uploaded_files_record.json'
    upload_files_to_drive(drive, directory_path, folder_id, record_file)