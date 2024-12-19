import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# App Title
st.title('Stock Price Predictor App')

# Add a banner image at the top
st.image(
    "C:\\Users\\Dell\\Desktop\\tf\\tfvenv\\S1.jpg",
    caption="Welcome to Stock Price Predictor",
    use_container_width=True
)

# Add a stock-related image in the sidebar
st.sidebar.image(
    "C:\\Users\\Dell\\Desktop\\tf\\tfvenv\\Stock_images1.png",
    caption="Stock Predictor",
    use_container_width=True
)

# Add an icon for the stock ticker input in the sidebar
# st.sidebar.image(
#     "C:\\Users\\Dell\\Desktop\\tf\\tfvenv\\S",
#     width=50  # Adjust the size of the icon
# )

# Load the pre-trained model
model = load_model("C:\\Users\\Dell\\Desktop\\tf\\tfvenv\\Latest_stock_price_model.keras")

# Define the date range
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Input for stock ticker
stock = st.text_input("Enter the stock ID", "GOOG")

# Download stock data
google_data = yf.download(stock, start, end)

# Flatten MultiIndex columns if they exist
if isinstance(google_data.columns, pd.MultiIndex):
    google_data.columns = [col[0] for col in google_data.columns]  # Flatten to single-level columns

# Handle missing data
if google_data.empty:
    st.error("No data found for the given stock ID. Please check the ticker symbol and try again.")
    st.stop()

# Ensure 'Close' column exists
if 'Close' not in google_data.columns:
    st.error("The 'Close' column is missing in the stock data. Please check the input or source data.")
    st.stop()

# Display the stock data
st.subheader("Stock Data")
st.write(google_data)

# Add a stock market-themed image above the data table
st.image(
    "C:\\Users\\Dell\\Desktop\\tf\\tfvenv\\S2.jpg",
    caption="Understanding the stock market trends",
    use_container_width=True
)

# Splitting the data
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data['Close'][splitting_len:])

# Define plot function
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset, 'g')
    return fig

# Moving Averages
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()

# Plot moving averages
st.subheader('Original Close Price and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

st.subheader('Original Close Price and MA for 200 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

st.subheader('Original Close Price and MA for 100 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

st.subheader('Original Close Price, MA for 100 days, and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, extra_data=1, extra_dataset=google_data['MA_for_250_days']))

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# Prepare test data for prediction
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

# Inverse scale predictions and test data
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare plotting data
ploting_data = pd.DataFrame(
    {
        'Original_test_data': inv_y_test.reshape(-1),  # means we are shaping it into 1D Array
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

# Display original vs predicted values
st.subheader("Original values vs predicted values")
st.write(ploting_data)

# Plot predictions vs original data
st.subheader("Original Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15, 6))
plt.plot(google_data['Close'][:splitting_len + 100], label="Data not used")
plt.plot(ploting_data['Original_test_data'], label="Original Test Data")
plt.plot(ploting_data['predictions'], label="Predicted Data")
plt.legend()
st.pyplot(fig)

# Add a footer caption
st.markdown(
    """
    ---
    **Thank you for using the Stock Price Predictor App!**  
    *Analyze stock trends and make informed decisions.*
    """
)



