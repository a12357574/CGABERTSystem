import streamlit as st
import pandas as pd
import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Authenticate with Google Drive using service account
def authenticate_drive():
    try:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive']
        )
        logger.info("Loaded service account credentials")
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"Failed to authenticate with Google Drive: {e}")
        st.stop()

# Global initialization
drive_service = authenticate_drive()

st.title("CGABERT Model Comparison with Google Colab")

# File upload
uploaded_file = st.file_uploader("Upload a CSV dataset for comparison", type=["csv"])
dataset_path = None

if uploaded_file is not None:
    # Save the uploaded file temporarily
    dataset_path = f"temp_{uploaded_file.name}"
    with open(dataset_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Validate CSV
    try:
        df = pd.read_csv(dataset_path)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column")
            os.remove(dataset_path)
            st.stop()
        st.success(f"Uploaded {uploaded_file.name} successfully")
    except Exception as e:
        st.error(f"Invalid CSV file: {e}")
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
        st.stop()

# Run comparison button
if st.button("Run Comparison"):
    if dataset_path is None:
        st.warning("Please upload a CSV file to run the comparison")
        st.stop()

    with st.spinner("Processing dataset..."):
        # Step 1: Upload CSV to Google Drive
        file_metadata = {'name': uploaded_file.name, 'parents': ['root']}
        media = MediaFileUpload(dataset_path)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        dataset_file_id = file.get('id')
        st.info(f"Uploaded dataset to Google Drive: {dataset_file_id}")

        # Move the file to the correct folder (cgabert_datasets)
        folder_metadata = {'name': 'cgabert_datasets', 'mimeType': 'application/vnd.google-apps.folder', 'parents': ['root']}
        folder_response = drive_service.files().list(q="name='cgabert_datasets' and mimeType='application/vnd.google-apps.folder' and trashed=false", fields='files(id)').execute()
        
        if 'files' in folder_response and folder_response['files']:
            folder_id = folder_response['files'][0]['id']
        else:
            folder = drive_service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = folder['id']
            st.info(f"Created new folder 'cgabert_datasets' with ID: {folder_id}")

        drive_service.files().update(fileId=dataset_file_id, addParents=folder_id, removeParents='root').execute()
        st.info(f"Moved dataset to cgabert_datasets folder")

        # Create or get cgabert_outputs folder
        output_folder = "cgabert_outputs"
        output_folder_response = drive_service.files().list(q=f"name='cgabert_outputs' and mimeType='application/vnd.google-apps.folder' and trashed=false", fields='files(id)').execute()
        
        if 'files' in output_folder_response and output_folder_response['files']:
            output_folder_id = output_folder_response['files'][0]['id']
            st.info(f"Found existing cgabert_outputs folder with ID: {output_folder_id}")
        else:
            folder_metadata = {'name': output_folder, 'mimeType': 'application/vnd.google-apps.folder', 'parents': ['root']}
            folder = drive_service.files().create(body=folder_metadata, fields='id').execute()
            output_folder_id = folder['id']
            st.info(f"Created new folder '{output_folder}' with ID: {output_folder_id}")

        st.write("Using output folder ID:", output_folder_id)

        # Step 2: Provide Colab Notebook Link with folder ID
        colab_link = "https://colab.research.google.com/drive/1FXErzFuhi5V5W1SbRiD6OJzFqLekfedb?usp=sharing"
        st.write("### Run the Model Comparison in Colab")
        st.write(f"Click the link below to open the Colab notebook, then click 'Run All' to execute the model comparison.")
        st.write(f"Ensure the dataset path in the notebook matches `/content/drive/My Drive/cgabert_datasets/{uploaded_file.name}`.")
        st.write(f"Also ensure the output folder in the notebook is set to `/content/drive/My Drive/cgabert_outputs` (Folder ID: `{output_folder_id}`).")
        st.markdown(f"[Open Colab Notebook]({colab_link})")
        st.info("After running the notebook, return here and click 'Check Results' to see the output.")
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

# Check results button
if st.button("Check Results"):
    with st.spinner("Fetching results from Google Drive..."):
        output_folder_id = "15jP1lX7Td7BGLixCvqKWw63ebpSNZElW"
        st.write("Using output folder ID:", output_folder_id)

        try:
            # --- START DEBUGGING BLOCK ---
            st.write("--- Debug Info ---")
            # List all files in the folder with detailed debugging
            all_files_query = f"'{output_folder_id}' in parents and trashed=false"
            st.write(f"Attempting to list all files with query: {all_files_query}")
            all_files_response = drive_service.files().list(q=all_files_query, fields='files(id, name, modifiedTime, owners, mimeType)').execute()
            if all_files_response.get('files'):
                st.write("Files found via general list query:")
                for f in all_files_response.get('files'):
                    st.write(f"- Name: {f.get('name')}, ID: {f.get('id')}, Modified: {f.get('modifiedTime')}, Owner: {f.get('owners', [{'displayName': 'N/A'}])[0].get('displayName')}, MIME Type: {f.get('mimeType')}")
            else:
                st.warning("General list query returned no files. This could be due to permissions, folder location, or API issue.")

            # Fetch results.csv (case-insensitive search)
            results_query = f"name contains 'results.csv' and '{output_folder_id}' in parents and trashed=false"
            st.write(f"Attempting to fetch results.csv with query: {results_query}")
            results_file = drive_service.files().list(q=results_query, fields='files(id, name, mimeType)').execute()
            if not results_file.get('files'):
                st.error("results.csv not found in cgabert_outputs folder. Check file name, permissions, or folder location.")
                st.stop()
            
            results_file_id = results_file['files'][0]['id']
            st.write(f"Found results file: {results_file['files'][0]['name']} (MIME Type: {results_file['files'][0].get('mimeType')})")
            request = drive_service.files().get_media(fileId=results_file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            with open("results.csv", "wb") as f:
                f.write(fh.getvalue())

            # Fetch interpretation.json (case-insensitive search)
            interpretation_query = f"name contains 'interpretation.json' and '{output_folder_id}' in parents and trashed=false"
            st.write(f"Attempting to fetch interpretation.json with query: {interpretation_query}")
            interpretation_file = drive_service.files().list(q=interpretation_query, fields='files(id, name, mimeType)').execute()
            if not interpretation_file.get('files'):
                st.error("interpretation.json not found in cgabert_outputs folder. Check file name, permissions, or folder location.")
                st.stop()

            interpretation_file_id = interpretation_file['files'][0]['id']
            st.write(f"Found interpretation file: {interpretation_file['files'][0]['name']} (MIME Type: {interpretation_file['files'][0].get('mimeType')})")
            request = drive_service.files().get_media(fileId=interpretation_file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            with open("interpretation.json", "wb") as f:
                f.write(fh.getvalue())

            st.write("--- End Debug Info ---")

            # Load and display results
            results_df = pd.read_csv("results.csv")
            with open("interpretation.json", "r") as f:
                interpretation = json.load(f)

            st.write("### Comparison Results")
            st.write(results_df)

            # Enhanced visualization with matplotlib
            st.write("### Model Performance Visualization")
            metrics_to_plot = ['Accuracy', 'F1_Score', 'Top_3_Accuracy', 'Top_5_Accuracy']
            if all(metric in results_df.columns for metric in metrics_to_plot):
                fig, ax = plt.subplots(figsize=(10, 6))
                bar_width = 0.2
                index = range(len(results_df['Model']))
                for i, metric in enumerate(metrics_to_plot):
                    plt.bar([j + i * bar_width for j in index], results_df[metric], bar_width, label=metric)
                plt.xlabel('Model')
                plt.ylabel('Score')
                plt.title('Model Performance Metrics')
                plt.xticks([i + bar_width * 1.5 for i in index], results_df['Model'])
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('performance_plot.png')
                st.image('performance_plot.png')
                os.remove('performance_plot.png')
            else:
                st.warning("Cannot create visualization: Required columns (Accuracy, F1_Score, Top_3_Accuracy, Top_5_Accuracy) missing in results.csv")

            st.write("### Interpretation")
            st.json(interpretation)

            # Clean up temporary files
            os.remove("results.csv")
            os.remove("interpretation.json")
            if os.path.exists(dataset_path):
                os.remove(dataset_path)

        except Exception as e:
            st.error(f"Error fetching results: {e}")
            if os.path.exists("results.csv"):
                os.remove("results.csv")
            if os.path.exists("interpretation.json"):
                os.remove("interpretation.json")
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
            st.stop()

else:
    st.info("Upload a CSV and click 'Run Comparison' to start")