from datetime import datetime, timedelta

import pandas as pd
import pandas_datareader as pdr
import plotly.graph_objects as go

CRYPTOS = ['BTC', 'ETH', 'LTC', 'XRP']
CURRENCY = 'GBP'

def getData(cryptocurrency):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    last_year_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")

    start = pd.to_datetime(last_year_date)
    end = pd.to_datetime(current_date)

    data = pdr.get_data_yahoo(f'{cryptocurrency}-{CURRENCY}', start, end)

    return data

crypto_data = {crypto:getData(crypto) for crypto in CRYPTOS} #dictionary comprehension

# crypto_data = dict()
# for crypto in CRYPTOS:
#     crypto_data[crypto] = getData(crypto)


fig = go.Figure()

# Scatter
for idx, name in enumerate(crypto_data):
    fig = fig.add_trace(
        go.Scatter(
            x = crypto_data[name].index,
            y = crypto_data[name].Close,
            name = name,
        )
    )

fig.update_layout(
    title = 'The Correlation between Different Cryptocurrencies',
    xaxis_title = 'Date',
    yaxis_title = f'Closing price ({CURRENCY})',
    legend_title = 'Cryptocurrencies'
)
fig.update_yaxes(type='log', tickprefix='£')

fig.show()