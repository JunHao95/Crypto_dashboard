import requests
import pandas as pd
import numpy as np

r = requests.get('https://api.alternative.me/fng/?limit=0')
print(r)
df = pd.DataFrame(r.json()['data'])

df.value = df.value.astype(int)
df.timestamp = pd.to_datetime(df.timestamp,unit='s')
print("Old df", df)
#exit()
#df.set_index('timestamp', inplace=True)
#print(df)

print(df.columns)
#df.plot(figsize=(20,10))
import plotly.graph_objects as go

colors = ["red" if val < 25 else "orange" if val <35 else "blue" if val <50  else "green" for val in df.value]
print("colors is ",colors)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = df.timestamp,
        y = df.value,
        mode="lines",
        name = " Fear & Greed",
        #marker=dict(color=colors)
        line = {'color': '#ff006a'},
        #line={'color': 'black'},
    )
)


fig.update_layout(
    title = "Fear and Greed Index for Crypto",
    xaxis_title = "Date",
    yaxis_title = "Greed Index",
    xaxis_rangeslider_visible = False
)
fig.show()

