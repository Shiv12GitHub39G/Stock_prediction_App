# Market Mind: Real Time Stock Forecast

This project is a **Market Mind: Real Time Stock Forecast** built using **Python** and the **Streamlit** framework. It leverages **Long Short-Term Memory (LSTM)** neural networks for stock price prediction and employs data analysis techniques to provide insights into stock market trends.

---

## Features

- **Stock Price Prediction**: Predicts stock prices for any company, including options like Google, Apple, and Microsoft.
- **Interactive Visualizations**: Includes charts and graphs to represent stock trends and prediction accuracy.
- **Customizable Inputs**: Users can specify the stock symbol to analyze and predict prices for any company listed on Yahoo Finance.
- **Low Prediction Error**: Utilized **Root Mean Squared Error (RMSE)** as the evaluation metric, achieving a low error value of **4**.

---

## Technologies Used

### Frameworks & Tools:
- **Streamlit**: For building the web app interface.
- **PyCharm**: IDE used for development.

### Machine Learning:
- **Keras Framework**:
  - **Sequential Model**
  - Layers: `Dense`, `LSTM`

### Data Processing & Visualization:
- **Libraries**:
  - `NumPy`
  - `Pandas`
  - `sklearn` (Scikit-learn)
  - `Matplotlib`

---

## Data Analysis

- Used historical stock data from **Yahoo Finance**.
- Analyzed trends and patterns in stock prices.
- Cleaned and preprocessed data for model training.

---

## Installation & Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/stock-predictor-web-app.git
   cd stock-predictor-web-app
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser:**
   The app will be accessible at `http://localhost:8501`.

---

## Future Enhancements

- Include additional machine learning models for comparison.
- Enhance the UI for a better user experience.
- Add more financial metrics and analysis tools.

---

## Acknowledgments

- **Yahoo Finance**: For providing the dataset.
- **Keras**: For simplifying the implementation of deep learning models.
- **Streamlit**: For enabling rapid development of web applications.

---

## Author

Developed by **Shivam Gupta**.
