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
import numpy as np

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

st.title("Improved Model Comparison with Google Colab")

# File upload
uploaded_file = st.file_uploader("Upload a CSV dataset for comparison", type=["csv"])
dataset_path = None

if uploaded_file is not None:
    dataset_path = f"temp_{uploaded_file.name}"
    with open(dataset_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
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
        file_metadata = {'name': uploaded_file.name, 'parents': ['root']}
        media = MediaFileUpload(dataset_path)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        dataset_file_id = file.get('id')
        st.info(f"Uploaded dataset to Google Drive: {dataset_file_id}")

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
            # Fetch results.csv
            results_query = f"name contains 'results.csv' and '{output_folder_id}' in parents and trashed=false"
            results_file = drive_service.files().list(q=results_query, fields='files(id, name, mimeType)').execute()
            if not results_file.get('files'):
                st.error("results.csv not found in cgabert_outputs folder.")
                st.stop()
            
            results_file_id = results_file['files'][0]['id']
            request = drive_service.files().get_media(fileId=results_file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            with open("results.csv", "wb") as f:
                f.write(fh.getvalue())

            # Fetch interpretation.json
            interpretation_query = f"name contains 'interpretation.json' and '{output_folder_id}' in parents and trashed=false"
            interpretation_file = drive_service.files().list(q=interpretation_query, fields='files(id, name, mimeType)').execute()
            if not interpretation_file.get('files'):
                st.error("interpretation.json not found in cgabert_outputs folder.")
                st.stop()

            interpretation_file_id = interpretation_file['files'][0]['id']
            request = drive_service.files().get_media(fileId=interpretation_file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            with open("interpretation.json", "wb") as f:
                f.write(fh.getvalue())

            # Load results
            results_df = pd.read_csv("results.csv")
            with open("interpretation.json", "r") as f:
                interpretation = json.load(f)

            # Create tabs for different sections
            tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Token Analysis", "Interpretation"])

            # Tab 1: Performance Metrics
            with tab1:
                st.write("### Model Performance Metrics")
                # Radar chart for multiple metrics
                metrics_to_plot = ['Accuracy', 'F1_Score', 'Top_3_Accuracy', 'Top_5_Accuracy']
                if all(metric in results_df.columns for metric in metrics_to_plot):
                    # Normalize Perplexity and Latency for radar chart (invert Perplexity and Latency so higher is better)
                    max_perplexity = max(results_df['Perplexity'])
                    results_df['Normalized_Perplexity'] = (max_perplexity - results_df['Perplexity']) / max_perplexity
                    max_latency = max(results_df['AvgLatencyMs'])
                    results_df['Normalized_Latency'] = (max_latency - results_df['AvgLatencyMs']) / max_latency

                    categories = metrics_to_plot + ['Normalized_Perplexity', 'Normalized_Latency']
                    values_base = results_df.loc[results_df['Model'] == 'BaseBERT', categories].values.flatten().tolist()
                    values_improved = results_df.loc[results_df['Model'] == 'Improved', categories].values.flatten().tolist()

                    # Repeat the first value to close the circle
                    values_base += values_base[:1]
                    values_improved += values_improved[:1]
                    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
                    angles += angles[:1]

                    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                    ax.fill(angles, values_base, color='blue', alpha=0.25, label='BaseBERT')
                    ax.fill(angles, values_improved, color='green', alpha=0.25, label='Improved')
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.set_title("Radar Chart: Model Performance Comparison")
                    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
                    plt.savefig('radar_chart.png')
                    st.image('radar_chart.png')
                    os.remove('radar_chart.png')
                else:
                    st.warning("Cannot create radar chart: Required columns missing.")

                # Display raw data
                st.write("### Raw Results")
                st.write(results_df)

            # Tab 2: Token Analysis
            with tab2:
                st.write("### Top Masked Tokens Analysis")
                base_tokens = interpretation['context']['base_metrics']['top_masked_tokens']
                improved_tokens = interpretation['context']['improved_metrics']['top_masked_tokens']

                # Prepare data for bar chart
                token_df = pd.DataFrame({
                    'Token': list(base_tokens.keys()) + list(improved_tokens.keys()),
                    'Frequency': list(base_tokens.values()) + list(improved_tokens.values()),
                    'Model': ['BaseBERT'] * len(base_tokens) + ['Improved'] * len(improved_tokens)
                })

                # Bar chart for top masked tokens
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = {'BaseBERT': 'blue', 'Improved': 'green'}
                for model in token_df['Model'].unique():
                    subset = token_df[token_df['Model'] == model]
                    ax.bar(subset['Token'], subset['Frequency'], label=model, color=colors[model], alpha=0.7)
                ax.set_xlabel('Token')
                ax.set_ylabel('Frequency')
                ax.set_title('Top Masked Tokens Comparison')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('token_chart.png')
                st.image('token_chart.png')
                os.remove('token_chart.png')

            # Tab 3: Interpretation
            with tab3:
                st.write("### Interpretation Summary")
                # Summary card for status and reason
                st.markdown(f"**Status**: {interpretation['status']}")
                st.markdown(f"**Reason**: {interpretation['reason']}")

                # Expander for detailed explanation
                with st.expander("Detailed Explanation"):
                    for key, value in interpretation['context']['explanation'].items():
                        st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

                # Expander for implications
                with st.expander("Implications and Next Steps"):
                    for key, value in interpretation['context']['implications'].items():
                        st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

            # Clean up
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