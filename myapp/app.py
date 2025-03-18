import yfinance as yf
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈", 
    layout="wide",  
)

def get_data(ticker, start_date, end_date):
    """Function to get stock data"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']


def create_features(data, lag):
    """Function to create lagged features"""
    lagged_features = pd.DataFrame(index=data.index)
    for company in data.columns:
        for i in range(1, lag + 1):
            lagged_features[f'{company}_lag_{i}'] = data[company].shift(i)
    return lagged_features


def train_model(X, Y):
    """Function to train a machine learning model"""
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    model = RandomForestRegressor()
    model.fit(X_imputed, Y)
    return model

#Sidebar
st.sidebar.header('⚙️ Settings')
st.sidebar.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/sosane-mahamoud-houssein/"
st.sidebar.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Sosane Mahamoud Houssein`</a>', unsafe_allow_html=True)

#User input: Add multiple companies
selected_stocks = st.sidebar.multiselect('Select Ticker Symbols:', ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'DIS', 'IBM', 'INTC', 'CSCO', 'C', 'CVX', 'PFE', 'KO', 'WMT', '^GSPC'])

#User input: Date range selector
start_date = st.sidebar.date_input('Select Start Date:', pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input('Select End Date:', pd.to_datetime('2025-01-01'))

#User input: Prediction horizon
prediction_horizon = st.sidebar.slider('Select Prediction Horizon (Days):', 1, 30, 7)

st.sidebar.markdown("---")
st.sidebar.markdown("### My Portfolio")
st.sidebar.link_button("Visit My Portfolio", "https://sosane.github.io/sosane-portfolio/")

#Display historical stock prices for selected companies
st.subheader('Historical Stock Prices')

#Get historical stock data for selected companies
combined_data = pd.DataFrame()

for stock in selected_stocks:
    try:
        selected_stock_data = get_data(stock, start_date, end_date)
        combined_data[stock] = selected_stock_data

        #Display historical stock prices using st.line_chart
        st.line_chart(selected_stock_data, use_container_width=True)

    except KeyError:
        st.warning(f"No 'Close' data found for {stock}. Please select a different stock.")

#Train a machine learning model on the combined dataset
if selected_stocks:
    lagged_features_all = create_features(combined_data[selected_stocks], lag=5)

    target_company = selected_stocks[0]  # Using the first company as a target variable
    target_variable = combined_data[target_company].shift(-1)

    data_for_modeling = pd.concat([lagged_features_all, target_variable], axis=1).dropna()

    X_train = data_for_modeling.drop(target_company, axis=1)
    Y_train = data_for_modeling[target_company]

    #Train the model
    model = train_model(X_train, Y_train)

    #Make predictions
    if st.button('Make Predictions'):
        #Get the latest available data for prediction
        latest_data = pd.DataFrame()

        for stock in selected_stocks:
            latest_stock_data = get_data(stock, end_date, end_date + pd.Timedelta(days=prediction_horizon))
            latest_data[stock] = latest_stock_data

        latest_lagged_features = create_features(latest_data[selected_stocks], lag=5)

        #Check if there's enough data for prediction
        if len(latest_lagged_features) < 1:
            st.warning(f"Not enough data available for prediction. Please select a longer date range.")
        else:
            #Create the imputer object before using it
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(X_train)

            #Use the fitted imputer to transform X_predict
            X_predict_imputed = pd.DataFrame(imputer.transform(latest_lagged_features), columns=X_train.columns)

            
            predictions = model.predict(X_predict_imputed) #Make predictions

            #Display the predicted prices for each selected company
            st.subheader('Predicted Prices for the Next {} Days:'.format(prediction_horizon))

            for i, stock in enumerate(selected_stocks):
                try:
                    st.write('{}: ${:.2f}'.format(stock, predictions[i]))
                except IndexError:
                    st.warning(f"No prediction available for {stock}. Please check the data range.")

            
            fig, ax = plt.subplots(figsize=(10, 6)) #Additional interactive visualization using Matplotlib

            for stock in selected_stocks:
                ax.plot(latest_data.index, latest_data[stock], label=stock)

            ax.set_title('Stock Prices Comparison')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price')
            ax.legend()

            #Format x-axis as dates
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

            st.pyplot(fig)
else:
    st.warning("Please select at least one Ticker Symbol.")

