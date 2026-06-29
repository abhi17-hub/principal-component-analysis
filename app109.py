import streamlit as st
import pandas as pd
import pickle

# Load Models
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("pca_model.pkl", "rb") as file:
    pca = pickle.load(file)

st.title("Principal Component Analysis (PCA)")
st.write("Upload data and transform it into Principal Components")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Original Data")
    st.write(df)

    # Scale Data
    scaled_data = scaler.transform(df)

    # PCA Transformation
    pca_data = pca.transform(scaled_data)

    pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])

    st.subheader("PCA Output")
    st.write(pca_df)

    csv = pca_df.to_csv(index=False)

    st.download_button(
        label="Download PCA Output",
        data=csv,
        file_name="pca_output.csv",
        mime="text/csv"
    )