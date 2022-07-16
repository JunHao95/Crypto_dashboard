import pandas as pd
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import numpy as np
import plotly.graph_objs as go
import pandas_datareader as pdr
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table
from datetime import datetime, timedelta
import requests
import json

df = pd.read_csv('token_new.csv')
columns = df.columns
for column in df:  # replace all empty cells with NaN
    # df[column].replace(' ',np.nan,inplace=True) #,should be NaN but try 0 to resolve bug
    df[column].replace(' ', 0, inplace=True)

# Analysing the data, access which columns are meaning and how to extract the data
categories = ["Price ($)", "TVL latest ($)", "Annualized GMV ($)", "7d protocol revenue ($)",
              "24h protocol revenue ($)", "30d protocol revenue ($)", "90d protocol revenue ($)",
              "180d protocol revenue ($)", "P/S ratio latest (x)", "P/E ratio latest (x)"]
print("There are {} number of columns we are actually interested in".format(len(categories)))

index = []  # get the index of the categories in the df columnns
for i in categories:
    index.append(df.columns.get_loc(i))
columns_index=[[5,15],[22,36],[37,49],[49,52],[52,55],[55,58],[58,61],[61,64],[64,78],[78,92]] #manually added in based on the input data, [[10], [14], [12], [2], [2], [2], [2], [2], [14], [14]]
coin_pair = list(zip(df.Project, df.Symbol))
coin_pair = [list(ele) for ele in coin_pair]

#Cheat sheet: https://dashcheatsheet.pythonanywhere.com/

#CSS component definition
tabs_styles = {
    'height': '60px',
    'align-items': 'center'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'border-radius': '15px',
    'background-color': '#F2F2F2',
    'box-shadow': '4px 4px 4px 4px lightgrey',

}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'border-radius': '15px',

}

header_style = {
    'borderBottom': '5px solid #d6d6d6',
    #'padding': '6px',
    'fontWeight': 'bold',
    'border-radius': '20px',
    'background-color': '#F2F2F2',
    'box-shadow': '4px 4px 4px 4px lightgrey',
    'textAlign': 'center',
    'text-transform': 'uppercase',
    'text-shadow': '2px 2px 4px Green',
}
title_style = {
    'color': 'Black',
    'fontWeight': "bold",
    'textAlign': 'center',
    'text-decoration': 'underline ',
    'text-shadow': '2px 2px 4px brown',
    'text-transform': 'uppercase',
}
graph_style ={
    "float":"mid",
}

#setting up crypto currency list for the graophs
url = "https://raw.githubusercontent.com/crypti/cryptocurrencies/master/cryptocurrencies.json"
resp = requests.get(url)
string = resp.text #string is in json format
formatted = json.loads(string)
cryptos_list = list(formatted.keys())
#Start for app definition
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets1 = [dbc.themes.DARKLY]
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale= 2, minimum-scale = 0.5"}],
    external_stylesheets=external_stylesheets,
)
server = app.server
app.title = "Crypto Information Dashboard (Outdated info)"


app.layout = html.Div([
    html.H1('Crypto Graph', style=header_style),
    dcc.Dropdown(
        id="crypto_dropdown",
        optionHeight=115,  # hard coded value for length of list
        options=[x for x in cryptos_list],
        value="BTC",
        multi=False
    ),

    html.Div([ # For Graph of crypto
        html.Div([
            dcc.Graph(id="graph_chart" ,style={'height':'70vh'} ),
        ]),
    ], className="row"),

    html.Br(),
    html.H1('Crypto Performance', style=header_style),
    dcc.Dropdown(
        id="dropdown",
        optionHeight=115, #hard coded value for length of list
        options=[x[0] for x in coin_pair],
        value="Bitcoin",
        multi=False
    ),
    html.Div([ # For price plot
        html.Div([
            dcc.Tabs(id="Bar_or_Table", value="Bar", children=[
                dcc.Tab(label="Bar Charts", value="Bar", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Tables", value="Table", style=tab_style, selected_style=tab_selected_style),
            ]),
        ]),
        html.Div([
            html.H3('Price change in Days', style=title_style),
            dcc.Graph(id="price_days"),
        ],className="five columns" ,style=graph_style),
        html.Div([
            html.H3('Price trend in Days', style=title_style),
            dcc.Graph(id="price_trends"),
        ],className="five columns" ,style=graph_style),
    ],className="row"),

    html.Div([ # For TVL plot
        html.Div([
            dcc.Tabs(id="Bar_or_Table1", value="Bar", children=[
                dcc.Tab(label="Bar Charts", value="Bar", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Tables", value="Table", style=tab_style, selected_style=tab_selected_style),
            ]),
        ]),
        html.Div([
            html.H3('TVL price change in Days', style=title_style),
            dcc.Graph(id="TVL_price_days"),
        ], className="five columns",style=graph_style),
        html.Div([
            html.H3('TVL trend in Days', style=title_style),
            dcc.Graph(id="TVL_price_trends"),
        ], className="five columns",style=graph_style),
        html.Div([
            html.H3('Avg TVL in Days', style=title_style),
            dcc.Graph(id="TVL_price_ranges"),
        ], className="five columns",style=graph_style),
    ],className="row"),

    html.Div([ #For Annualized plot
        html.Div([
            dcc.Tabs(id="Bar_or_Table2", value="Bar", children=[
                dcc.Tab(label="Bar Charts", value="Bar", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Tables", value="Table", style=tab_style, selected_style=tab_selected_style),
            ]),
        ],className="twelve columns"),
        html.Div([
            html.H3('Annualized Revenue in days ', style=title_style),
            dcc.Graph(id="Annualized_revenue"),
        ], className="five columns",style=graph_style),
        html.Div([
            html.H3('Annualized Price change in days ', style=title_style),
            dcc.Graph(id="Annualized_price_days"),
        ], className="five columns",style=graph_style),
        html.Div([
            html.H3('Annualized Price trend in days ', style=title_style),
            dcc.Graph(id="Annualized_price_trend", style={"align-items":"center"}),
        ], className="five columns",style=graph_style)
    ]),
    html.Div([
        html.Div([
            dcc.Tabs(id="Bar_or_Table3", value="Bar", children=[
                dcc.Tab(label="Bar Charts", value="Bar", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Tables", value="Table", style=tab_style, selected_style=tab_selected_style),
            ]),
        ],className="twelve columns"),
        html.Div([ #For Protocol Revenue
            html.H3('Protocol 7 days change ', style=title_style),
            dcc.Graph(id="Protocol_1"),
        ], className="five columns",style=graph_style),
        html.Div([
            html.H3('Protocol 1 days change ', style=title_style),
            dcc.Graph(id="Protocol_2"),
        ], className="five columns",style=graph_style),
        html.Div([
            html.H3('Protocol 30 days change ', style=title_style),
            dcc.Graph(id="Protocol_3"),
        ], className="five columns",style=graph_style),
        html.Div([
            html.H3('Protocol 90 days change ', style=title_style),
            dcc.Graph(id="Protocol_4"),
        ], className="five columns",style=graph_style),
        html.Div([
            html.H3('Protocol 180 days change ', style=title_style),
            dcc.Graph(id="Protocol_5"),
        ], className="five columns", style=graph_style)
    ]),
    html.Div([#For P_S ratio
        html.Div([
            dcc.Tabs(id="Bar_or_Table4", value="Bar", children=[
                dcc.Tab(label="Bar Charts", value="Bar", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Tables", value="Table", style=tab_style, selected_style=tab_selected_style),
            ]),
        ],className="twelve columns"),
        html.Div([
            html.H3('P/S Ratio ', style=title_style),
            dcc.Graph(id="PS_ratio"),
        ], className="five columns", style=graph_style),
        html.Div([
            html.H3('Avg P/S Ratio ', style=title_style),
            dcc.Graph(id="PS_avg_ratio"),
        ], className="five columns", style=graph_style)
    ]),
    html.Div([ # For P_E ratio
        html.Div([
            dcc.Tabs(id="Bar_or_Table5", value="Bar", children=[
                dcc.Tab(label="Bar Charts", value="Bar", style=tab_style, selected_style=tab_selected_style,),
                dcc.Tab(label="Tables", value="Table", style=tab_style, selected_style=tab_selected_style),
            ]),
        ],className="twelve columns"),
        html.Div([
            html.H3('P/E Ratio ', style=title_style),
            dcc.Graph(id="PE_ratio"),
        ], className="five columns", style=graph_style),
        html.Div([
            html.H3('Avg P/E Ratio ', style=title_style),
            dcc.Graph(id="PE_avg_ratio"),
        ], className="five columns", style=graph_style)
    ])
], id="container")


@app.callback(
    Output("graph_chart", "figure"),
    [Input("crypto_dropdown", "value")]) # based on the id of the dropdown
def crypto_graph(value):
    print("old Value is ",value)
    value = str(value)
    #value = [str(cry) for cry in value ]
    CURRENCY = 'USD'  # all crypto values are in USD
    # Getting data for the crypto
    print("Value is ",value)
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    last_year_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    start = pd.to_datetime(last_year_date)
    end = pd.to_datetime(current_date)
    crypto_data = pdr.get_data_yahoo(f'{value}-{CURRENCY}', start, end)
    #crypto_data = {crypto: getData(crypto) for crypto in value}  # dictionary comprehension
    print("crypto data is ",crypto_data)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=crypto_data.index,
                open=crypto_data.Open,
                high=crypto_data.High,
                low=crypto_data.Low,
                close=crypto_data.Close
            ),
            go.Scatter(
                x=crypto_data.index,
                y=crypto_data.Close.rolling(window=20).mean(),
                mode='lines',
                name='20SMA',
                line={'color': '#ff006a'}
            ),
            go.Scatter(
                x=crypto_data.index,
                y=crypto_data.Close.rolling(window=50).mean(),
                mode='lines',
                name='50SMA',
                line={'color': '#1900ff'}
            )
        ]
    )

    fig.update_layout(
        title=f'The Candlestick graph for {value}',
        title_x = 0.5,
        xaxis_title='Date',
        yaxis_title=f'Price ({CURRENCY})',
        xaxis_rangeslider_visible=False,
        legend=dict(yanchor="top",xanchor="right", y=0.99,x=0.99)
    )
    fig.update_yaxes(tickprefix='$')

    return fig

@app.callback(
    Output("price_days", "figure"),
    Output("price_trends", "figure"),
    [Input("dropdown", "value"), Input("Bar_or_Table", "value")]) # based on the id of the dropdown
def update_bar_chart(value,value1):
    data = df
    idx_range = columns_index[0]
    days_range = [idx_range[0], idx_range[0] + 5]
    trend_range = [days_range[1], idx_range[1]]
    idx = data.index[data["Project"] == value].tolist()  # retrieve the row that belongs to the specific value
    data = data.iloc[idx, :]
    data.reset_index(drop=True, inplace=True)
    df_days = data.iloc[:, [x for x in range(days_range[0], days_range[1])]]
    df_trend = data.iloc[:, [x for x in range(trend_range[0], trend_range[1])]]
    df_days = df_days.transpose().reset_index()
    df_days.rename(columns={"index": "Price_days", 0: "Value"}, inplace=True)
    df_trend = df_trend.transpose().reset_index()
    df_trend.rename(columns={"index": "Price_trends", 0: "Value"}, inplace=True)
    if value1 == "Table":
        fig1 = go.Figure(data=[go.Table(
            columnwidth=[400,200],
            header=dict(values=list(df_days.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_days.Price_days, df_days.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig2 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_trend.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_trend.Price_trends, df_trend.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        return fig1, fig2
    else:
        px1 = px.bar(df_days, x="Price_days", y="Value")
        px2 = px.bar(df_trend, x="Price_trends", y="Value")
        px1.update_layout(autosize=False)
        px2.update_layout(autosize=False)
        return px1, px2



@app.callback(
    Output("TVL_price_days", "figure"),
    Output("TVL_price_trends", "figure"),
    Output("TVL_price_ranges", "figure"),
    [Input("dropdown", "value"), Input("Bar_or_Table1", "value")])  # based on the id of the dropdown
def TVL_plot(value,value1):
    """Take in that dataframe and value type and filter for only price related data and columns. there are days change and days trend change"""
    data =df
    idx_range = columns_index[1]
    days_range = [idx_range[0], idx_range[0]+5]
    trend_range = [days_range[1],days_range[1]+5]
    avg_range = [trend_range[1],idx_range[1]]
    idx = data.index[data["Project"] == value].tolist() #retrieve the row that belongs to the specific value
    data = data.iloc[idx,:]
    data.reset_index(drop=True, inplace=True)
    df_days = data.iloc[:,[x for x in range(days_range[0],days_range[1])]]
    df_trend = data.iloc[:,[x for x in range(trend_range[0],trend_range[1])]]
    df_range = data.iloc[:,[x for x in range(avg_range[0],avg_range[1])]]
    df_days = df_days.transpose().reset_index()
    df_days.rename(columns={"index":"Price_days", 0: "Value"},inplace=True)
    df_trend = df_trend.transpose().reset_index()
    df_trend.rename(columns={"index":"Price_trends", 0: "Value"},inplace=True)
    df_range = df_range.transpose().reset_index()
    df_range.rename(columns={"index":"Price_ranges", 0: "Value"},inplace=True)
    if value1 == "Table":
        fig1 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_days.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_days.Price_days, df_days.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig2 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_trend.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_trend.Price_trends, df_trend.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig3 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_range.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_range.Price_ranges, df_range.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        return fig1, fig2, fig3
    else:
        fig1 = px.bar(df_days, x="Price_days", y="Value")
        fig2 = px.bar(df_trend, x="Price_trends", y="Value")
        fig3 = px.bar(df_range, x="Price_ranges", y="Value")
        fig1.update_layout()
        fig2.update_layout()
        fig3.update_layout()
        return fig1,fig2,fig3
@app.callback(
    Output("Annualized_revenue", "figure"),
    Output("Annualized_price_days", "figure"),
    Output("Annualized_price_trend", "figure"),
    [Input("dropdown", "value"), Input("Bar_or_Table2", "value")])  # based on the id of the dropdown
def Annualized_plot(value,value1):# 4,4,4
    """Take in that dataframe and value type and filter for only price related data and columns. there are days change and days trend change"""
    data =df
    idx_range = columns_index[2]
    revenues_range = [idx_range[0], idx_range[0] + 4]
    days_range = [revenues_range[1], revenues_range[1]+4]
    trend_range = [days_range[1],idx_range[1]]
    idx = data.index[data["Project"] == value].tolist() #retrieve the row that belongs to the specific value
    data = data.iloc[idx,:]
    data.reset_index(drop=True, inplace=True)
    df_revenue = data.iloc[:, [x for x in range(revenues_range[0], revenues_range[1])]]
    df_days = data.iloc[:,[x for x in range(days_range[0],days_range[1])]]
    df_trend = data.iloc[:,[x for x in range(trend_range[0],trend_range[1])]]
    df_revenue = df_days.transpose().reset_index()
    df_revenue.rename(columns={"index": "Revenue_days", 0: "Value"}, inplace=True)
    df_days = df_days.transpose().reset_index()
    df_days.rename(columns={"index":"Price_days" , 0: "Value"},inplace=True)
    df_trend = df_trend.transpose().reset_index()
    df_trend.rename(columns={"index":"Price_trends" , 0: "Value"},inplace=True)
    if value1 == "Table":
        fig1 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_revenue.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_revenue.Revenue_days, df_revenue.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig2 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_days.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_days.Price_days, df_days.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig3 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_trend.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_trend.Price_trends, df_trend.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        return fig1, fig2, fig3
    else:
        fig1 = px.bar(df_revenue, x=df_revenue["Revenue_days"], y=df_revenue["Value"])
        fig2 = px.bar(df_days, x=df_days["Price_days"], y=df_days["Value"])
        fig3 = px.bar(df_trend, x=df_trend["Price_trends"], y=df_trend["Value"])
        fig1.update_layout()
        fig2.update_layout()
        fig3.update_layout()
        return fig1,fig2,fig3

@app.callback(
    Output("Protocol_1", "figure"),
    Output("Protocol_2", "figure"),
    Output("Protocol_3", "figure"),
    Output("Protocol_4", "figure"),
    Output("Protocol_5", "figure"),
    [Input("dropdown", "value"), Input("Bar_or_Table3", "value")]) # based on the id of the dropdown
def protocol_revenue_plot(value,value1): #3,3,3,3,3
    """Take in that dataframe and value type and filter for only price related data and columns. there are days change and days trend change"""
    data =df
    idx_range = columns_index[3:8]
    range1 = [idx_range[0][0], idx_range[0][1]]
    range2 = [idx_range[1][0], idx_range[1][1]]
    range3 = [idx_range[2][0], idx_range[2][1]]
    range4 = [idx_range[3][0], idx_range[3][1]]
    range5 = [idx_range[4][0], idx_range[4][1]]
    idx = data.index[data["Project"] == value].tolist()  # retrieve the row that belongs to the specific value
    data = data.iloc[idx, :]
    data.reset_index(drop=True, inplace=True)
    df_range1 = data.iloc[:, [x for x in range(range1[0], range1[1])]]
    df_range2 = data.iloc[:, [x for x in range(range2[0], range2[1])]]
    df_range3 = data.iloc[:, [x for x in range(range3[0], range3[1])]]
    df_range4 = data.iloc[:, [x for x in range(range4[0], range4[1])]]
    df_range5 = data.iloc[:, [x for x in range(range5[0], range5[1])]]
    df_range1 = df_range1.transpose().reset_index()
    df_range1.rename(columns={"index": "Protocol_1", 0: "Value"}, inplace=True)
    df_range2 = df_range2.transpose().reset_index()
    df_range2.rename(columns={"index": "Protocol_2", 0: "Value"}, inplace=True)
    df_range3 = df_range3.transpose().reset_index()
    df_range3.rename(columns={"index": "Protocol_3", 0: "Value"}, inplace=True)
    df_range4 = df_range4.transpose().reset_index()
    df_range4.rename(columns={"index": "Protocol_4", 0: "Value"}, inplace=True)
    df_range5 = df_range5.transpose().reset_index()
    df_range5.rename(columns={"index": "Protocol_5", 0: "Value"}, inplace=True)
    if value1 == "Table":
        fig1 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_range1.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_range1.Protocol_1, df_range1.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig2 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_range2.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_range2.Protocol_2, df_range2.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig3 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_range3.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_range3.Protocol_3, df_range3.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig4 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_range4.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_range4.Protocol_4, df_range4.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig5 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_range5.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_range5.Protocol_5, df_range5.Value],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        return fig1, fig2,fig3,fig4,fig5
    else:
        fig1 = px.bar(df_range1, x=df_range1["Protocol_1"], y=df_range1["Value"])
        fig2 = px.bar(df_range2, x=df_range2["Protocol_2"], y=df_range2["Value"])
        fig3 = px.bar(df_range3, x=df_range3["Protocol_3"], y=df_range3["Value"])
        fig4 = px.bar(df_range4, x=df_range4["Protocol_4"], y=df_range4["Value"])
        fig5 = px.bar(df_range5, x=df_range5["Protocol_5"], y=df_range5["Value"])
        fig1.update_layout()
        fig2.update_layout()
        fig3.update_layout()
        fig4.update_layout()
        fig5.update_layout()
        return fig1,fig2,fig3,fig4,fig5

@app.callback(
    Output("PS_ratio", "figure"),
    Output("PS_avg_ratio", "figure"),
    [Input("dropdown", "value"), Input("Bar_or_Table4", "value")])  # based on the id of the dropdown
def P_S_ratio_plot(value,value1): #11,4
    """Take in that dataframe and value type and filter for only price related data and columns. there are days change and days trend change"""
    data =df
    idx_range = columns_index[8]
    ps_ratio_range = [idx_range[0], idx_range[0] + 10]
    ps_ratio_day_range = [ps_ratio_range[1], idx_range[1]]
    idx = data.index[data["Project"] == value].tolist()  # retrieve the row that belongs to the specific value
    data = data.iloc[idx, :]
    data.reset_index(drop=True, inplace=True)
    df_ratio = data.iloc[:, [x for x in range(ps_ratio_range[0], ps_ratio_range[1])]]
    df_avg_ratio = data.iloc[:, [x for x in range(ps_ratio_day_range[0], ps_ratio_day_range[1])]]
    df_ratio = df_ratio.transpose().reset_index()
    df_ratio.rename(columns={"index": "P/S ratio", 0: "Value"}, inplace=True)
    df_avg_ratio = df_avg_ratio.transpose().reset_index()
    df_avg_ratio.rename(columns={"index": "Avg P/S ratio", 0: "Value"}, inplace=True)
    if value1 == "Table":
        fig1 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_ratio.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_ratio["P/S ratio"], df_ratio["Value"]],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig2 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_avg_ratio.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_avg_ratio["Avg P/S ratio"], df_avg_ratio["Value"]],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        return fig1, fig2
    else:
        fig1 = px.bar(df_ratio, x=df_ratio["P/S ratio"], y=df_ratio["Value"])
        fig2 = px.bar(df_avg_ratio, x=df_avg_ratio["Avg P/S ratio"], y=df_avg_ratio["Value"])
        fig1.update_layout()
        fig2.update_layout()
        return fig1, fig2

@app.callback(
    Output("PE_ratio", "figure"),
    Output("PE_avg_ratio", "figure"),
    [Input("dropdown", "value"), Input("Bar_or_Table5", "value")])  # based on the id of the dropdown
def P_E_ratio_plot(value,value1):
    """Take in that dataframe and value type and filter for only price related data and columns. there are days change and days trend change"""
    data =df
    idx_range = columns_index[9]
    pe_ratio_range = [idx_range[0], idx_range[0] + 10]
    pe_ratio_day_range = [pe_ratio_range[1], idx_range[1]]
    idx = data.index[data["Project"] == value].tolist()  # retrieve the row that belongs to the specific value
    data = data.iloc[idx, :]
    data.reset_index(drop=True, inplace=True)
    df_ratio = data.iloc[:, [x for x in range(pe_ratio_range[0], pe_ratio_range[1])]]
    df_avg_ratio = data.iloc[:, [x for x in range(pe_ratio_day_range[0], pe_ratio_day_range[1])]]
    df_ratio = df_ratio.transpose().reset_index()
    df_ratio.rename(columns={"index": "P/E ratio", 0: "Value"}, inplace=True)
    df_avg_ratio = df_avg_ratio.transpose().reset_index()
    df_avg_ratio.rename(columns={"index": "Avg P/E ratio", 0: "Value"}, inplace=True)
    if value1 == "Table":
        fig1 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_ratio.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_ratio["P/E ratio"], df_ratio["Value"]],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        fig2 = go.Figure(data=[go.Table(
            columnwidth=[400, 200],
            header=dict(values=list(df_avg_ratio.columns),
                        fill_color='paleturquoise',
                        font_size=20,
                        height=40,
                        align='center'),
            cells=dict(values=[df_avg_ratio["Avg P/E ratio"], df_avg_ratio["Value"]],
                       fill_color='lavender',
                       font_size=15,
                       height=30,
                       align='center'))
        ])
        return fig1, fig2
    else:
        fig1 = px.bar(df_ratio, x=df_ratio["P/E ratio"], y=df_ratio["Value"])
        fig2 = px.bar(df_avg_ratio, x=df_avg_ratio["Avg P/E ratio"], y=df_avg_ratio["Value"])
        fig1.update_layout()
        fig2.update_layout()
        return fig1, fig2


if __name__ == '__main__':
    app.run_server(debug=True, port=8056)

