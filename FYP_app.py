
import streamlit as st
import pandas as pd
from hybrid_model import run_hybrid_forecast

st.title("📊 Stock Trends Forecasting and Visualization (FYP)")

ticker = st.sidebar.selectbox("Select Stock", ["AAPL", "SAP", "005930.KS"], index=0)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2020-12-31"))
epochs = st.sidebar.slider("Epochs for LSTM", 1, 20, value=5)

if st.button("Run Forecast"):
    with st.spinner("Training model... please wait"):
        fig, mse, df, error = run_hybrid_forecast(ticker, start_date, end_date, epochs=epochs)

    if error:
        st.error(error)
    else:
        st.subheader(f"{ticker} Stock Data")
        st.line_chart(df['Close'])
        st.subheader("Hybrid Model Forecast Result")
        st.pyplot(fig)
        st.metric(label="Mean Squared Error", value=f"{mse:.4f}")
