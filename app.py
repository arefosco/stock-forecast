# libraries import
import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go




# initial and end date
I_DATE = '2017-01-01'
E_DATE = date.today().strftime('%Y-%m-%d')

st.title('Stocks Analysis')

# Sidebar creation

st.sidebar.header('Chose the stock')

n_days = st.slider('Number of forecasting days', 365, 1095)


# stocks id's
def stocks_data_collect():
    path = 'C:/Users/AndreRefosco/app-acoes/acoes.csv'
    return pd.read_csv(path, delimiter=';')


df = stocks_data_collect()

stock = df['snome']

chosen_stock_name = st.sidebar.selectbox('Choose one stock', stock)

df_stock = df[df['snome'] == chosen_stock_name]
chosen_stock = df_stock.iloc[0]['sigla_acao']
# chosen_stock = chosen_stock + '.SA'


@st.cache_data
def get_values_online(chosen_stock_name):
    df = yf.download(chosen_stock_name, I_DATE, E_DATE)
    df.reset_index(inplace=True)
    return df


df_values = get_values_online(chosen_stock)
st.subheader('Prices Table -' + chosen_stock_name)
st.write(df_values.tail(10))

# Graph creating

st.subheader('Prices Graph')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_values['Date'],
                         y=df_values['Close'],
                         name='Close Price',
                         line_color = 'blue'))

fig.add_trace(go.Scatter(x=df_values['Date'],
                         y=df_values['Open'],
                         name='Openn Price',
                         line_color = 'green'))

st.plotly_chart(fig)

# Forecasting

df_train = df_values[['Date', 'Close']]

# Rename columns

df_train = df_train.rename(columns={"Date": 'ds', "Close": 'y'})

model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods=n_days, freq='B')

forecast = model.predict(future)
st.subheader('Forecast')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days))


# graph
graph1 = plot_plotly(model, forecast)
st.plotly_chart(graph1)

# graph2
graph2 = plot_components_plotly(model, forecast)
st.plotly_chart(graph2)
