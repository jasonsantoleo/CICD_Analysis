import streamlit as st
import pandas as pd
from typing import List, Dict, Any

import Gemini_utils
# Page configuration
st.set_page_config(page_title="CI/CD Log Anomalies", layout="wide")
st.title("🔍 CI/CD Log Anomalies")

# Sidebar controls
threshold = st.sidebar.slider(
    "Anomaly probability threshold",
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.01
)

# File uploader
uploaded = st.file_uploader(
    "Upload one or more JSON log files",
    type="json",
    accept_multiple_files=True
)

if uploaded:
    # Process uploaded files
    dfs = []
    for f in uploaded:
        try:
            dfs.append(pd.read_json(f))
        except ValueError:
            st.warning(f"Could not read {f.name}")
    
    # Check if any files were successfully read
    if not dfs:
        st.error("No valid JSON files found")
        st.stop()
    
    # Combine all dataframes
    data = pd.concat(dfs, ignore_index=True)
    st.sidebar.metric("Total records", len(data))
    
    # Convert to list of dictionaries for processing
    records: List[Dict[str, Any]] = data.to_dict(orient="records")
    
    # Detect anomalies using OpenAI utils
    anomalies = Gemini_utils.detect_anomalies(records, threshold=threshold)
    n_anom = len(anomalies)
    st.sidebar.metric("Anomalies found", n_anom)
    
    if n_anom:
        # Display anomalies
        df_anom = pd.DataFrame(anomalies)
        df_anom = df_anom.sort_values("anomaly_prob", ascending=False)
        
        st.subheader("List of potential anomalies")
        st.dataframe(
            df_anom[[
                "anomaly_prob", "run_id", "stage", "status", "timestamp", "message"
            ]],
            use_container_width=True
        )
        
        # Generate anomaly description
        if st.button("Describe anomalies"):
            with st.spinner("Generating overview..."):
                try:
                    summary = Gemini_utils.describe_anomalies(anomalies)
                    st.markdown("**Anomaly overview (2-3 sentences):**")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error calling OpenAI: {e}")
    else:
        st.info("No anomalies found above threshold")
    
    # Prepare download data
    df_all = data.copy()
    df_all["anomaly_prob"] = 0.0
    
    # Add anomaly probabilities to the full dataset
    for rec in anomalies:
        mask = (
            (df_all["run_id"] == rec["run_id"]) &
            (df_all["timestamp"].astype(str) == str(rec["timestamp"])) &
            (df_all["stage"] == rec["stage"])
        )
        df_all.loc[mask, "anomaly_prob"] = rec["anomaly_prob"]
    
    # Download button
    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results (all records with probabilities)",
        data=csv,
        file_name="cicd_anomaly_results.csv",
        mime="text/csv"
    )
else:
    st.info("Upload JSON log files for analysis")