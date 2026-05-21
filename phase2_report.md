## 1. System Components & Interactions

* **Users:** Submit stock tickers (e.g., AAPL) and view predicted trends via the UI.
* **Frontend / Interface (TypeScript / React):** The user-facing dashboard. It handles input validation and visualizes the stock charts and ML predictions.
* **Backend Services (FastAPI):** The central orchestrator. It receives user requests, handles business logic, triggers data fetching, and routes data to the ML engine for inference.
* **External APIs (yfinance):** Acts as the primary data source for historical market data (OHLCV) and company metadata.
* **Data Pipelines:** A scheduled or trigger-based Python script that cleans raw `yfinance` data (handling missing values, calculating moving averages, scaling) before passing it to the database or model.
* **Databases (PostgreSQL):** Stores cleaned stock data, user configurations, and historical prediction logs.
* **AI / ML Components:** * **Training Pipeline:** Fetches historical data from the DB to train and validate the model.
    * **Inference Engine:** Loads the trained model and returns real-time predictions based on current market data.

---

## 2. Data Flow

1.  **Request:** User queries a specific stock ticker on the Frontend.
2.  **API Call:** Frontend sends a GET request to the Backend.
3.  **Data Fetch & Process:** Backend checks the Database for recent data. If outdated, it triggers the Data Pipeline to fetch fresh data from `yfinance`, preprocess it, and update the DB.
4.  **Inference:** Backend sends the preprocessed current data to the ML Inference Engine.
5.  **Response:** The engine returns the prediction to the Backend, which serves it back to the Frontend for rendering.

---

## 3. Main Component Contracts

* **Frontend ↔ Backend:** Standard RESTful API communicating via JSON payloads (e.g., `{"ticker": "AAPL", "timeframe": "1d"}`).
* **Backend ↔ yfinance API:** Python library abstraction executing HTTP requests, returning standard Pandas DataFrames or JSON dictionaries.
* **Backend ↔ ML Inference:** Function calls or internal microservice requests passing normalized feature arrays and receiving specific output tensors/values.

---

## 4. Machine Learning Pipeline

### Recommended Model
For time-series stock forecasting, the primary architecture is an **LSTM (Long Short-Term Memory)** neural network. LSTMs are specifically designed to retain long-term dependencies in sequential data, making them ideal for historical price trends. *(Note: XGBoost can be used as a faster, non-deep-learning baseline).*

### Training Process

1.  **Fetch & Engineer Features**
    * Pull OHLCV (Open, High, Low, Close, Volume) data via `yfinance`.
    * Add technical indicators to the dataset (e.g., Moving Averages, RSI, MACD).

2.  **Preprocess & Scale**
    * Normalize all features using `MinMaxScaler` to bind values between 0 and 1, as financial data varies wildly in scale.

3.  **Create Sequences (Windowing)**
    * Group the data into rolling windows to create 3D input arrays: `(samples, time_steps, features)`.
    * *Example:* Use the features from the past 60 days to predict the target for day 61.

4.  **Build the Architecture**
    * Initialize a Sequential model (TensorFlow/Keras or PyTorch).
    * Add 1–2 LSTM layers with Dropout (e.g., 0.2) to prevent overfitting.
    * Add a final Dense (Linear) output layer consisting of 1 neuron.

5.  **Compile & Optimize**
    * **Loss Function:** Mean Squared Error (MSE).
    * **Optimizer:** Adam (efficiently handles gradient descent for noisy financial data).
    * **Train:** Split the data chronologically (e.g., train on 80% older data, test on 20% recent data). Never randomize or shuffle time-series splits.