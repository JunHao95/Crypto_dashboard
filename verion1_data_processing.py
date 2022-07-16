import pandas as pd
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

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
#print("Done getting coin pair of lenght,", len(coin_pair))
#print(coin_pair)





#Start for app definition
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.CYBORG],
)
server = app.server
app.title = "Crypto Information Dashboard (Outdated info)"


app.layout = html.Div([
    html.H1('Crypto Performance'),
    html.Hr(),
    dcc.Dropdown(
        id="dropdown",
        optionHeight=115, #hard coded value for length of list
        options=[ x[0] for x in coin_pair],
        value="Cardano",
        multi=False
    ),
    html.Div([
        dcc.Graph(id="price_days"),
        # dcc.Graph(id="TVL"),
        # dcc.Graph(id="Annualized"),
        # dcc.Graph(id="Protocol_Revenue"),
        # dcc.Graph(id="P_S_Ratio"),
        # dcc.Graph(id="P_E_Ratio"),
    ],className="two columns"),
    html.Div([
        dcc.Graph(id="price_trends"),
    ],className="two columns")
], id="container")

@app.callback(
    Output("price_days", "figure"),
    Output("price_trends", "figure"),
    # Output("TVL", "figure"),
    # Output("Annualized", "figure"),
    # Output("Protocol_Revenue", "figure"),
    # Output("P_S_Ratio", "figure"),
    # Output("P_E_Ratio", "figure"),
    [Input("dropdown", "value")]) # based on the id of the dropdown
def update_bar_chart(value):
    import plotly.graph_objects as go
    category = categories[0]
    data = df
    idx_range = columns_index[0]
    days_range = [idx_range[0], idx_range[0] + 5]
    trend_range = [days_range[1], idx_range[1]]
    idx = data.index[data["Project"] == value].tolist()  # retrieve the row that belongs to the specific coin
    data = data.iloc[idx, :]
    data.reset_index(drop=True, inplace=True)
    df_days = data.iloc[:, [x for x in range(days_range[0], days_range[1])]]
    df_trend = data.iloc[:, [x for x in range(trend_range[0], trend_range[1])]]
    df_days = df_days.transpose().reset_index()
    df_days.rename(columns={"index": "Price_days", 0: "Value"}, inplace=True)
    df_trend = df_trend.transpose().reset_index()
    df_trend.rename(columns={"index": "Price_trends", 0: "Value"}, inplace=True)
        # fig1 = px.scatter(df_days,x=df_days["Price_days"],y=df_days["Value"])
        # fig2 = px.scatter(df_trend,x=df_trend["Price_trends"],y=df_trend["Value"])
        # fig1 = go.Figure(data=[go.Bar(x=[df_days["Price_days"]],y=[df_days["Value"]])])
        # fig2 = go.Figure(data=[go.scatter(x=df_trend["Price_trends"],y=df_trend["Value"])])
    px1 = px.line(df_days, x="Price_days", y="Value")
    px2 = px.line(df_trend, x="Price_trends", y="Value")
    px1.update_layout()
    px2.update_layout()
    return px1, px2



    # def price_plot(data, coin, idx_range):
    #     """Take in that dataframe and coin type and filter for only price related data and columns. there are days change and days trend change"""
    #     import plotly.graph_objects as go
    #     category = categories[0]
    #     days_range = [idx_range[0], idx_range[0] + 5]
    #     trend_range = [days_range[1], idx_range[1]]
    #     idx = data.index[data["Project"] == coin].tolist()  # retrieve the row that belongs to the specific coin
    #     data = data.iloc[idx, :]
    #     data.reset_index(drop=True, inplace=True)
    #     df_days = data.iloc[:, [x for x in range(days_range[0], days_range[1])]]
    #     df_trend = data.iloc[:, [x for x in range(trend_range[0], trend_range[1])]]
    #     df_days = df_days.transpose().reset_index()
    #     df_days.rename(columns={"index": "Price_days", 0: "Value"}, inplace=True)
    #     df_trend = df_trend.transpose().reset_index()
    #     df_trend.rename(columns={"index": "Price_trends", 0: "Value"}, inplace=True)
    #     # fig1 = px.scatter(df_days,x=df_days["Price_days"],y=df_days["Value"])
    #     # fig2 = px.scatter(df_trend,x=df_trend["Price_trends"],y=df_trend["Value"])
    #     # fig1 = go.Figure(data=[go.Bar(x=[df_days["Price_days"]],y=[df_days["Value"]])])
    #     # fig2 = go.Figure(data=[go.scatter(x=df_trend["Price_trends"],y=df_trend["Value"])])
    #     px1 = px.line(df_days, x="Price_days", y="Value")
    #     px2 = px.line(df_trend, x="Price_trends", y="Value")
    #     return px1, px2

# def TVL_plot(data,coin,idx_range):
#     """Take in that dataframe and coin type and filter for only price related data and columns. there are days change and days trend change"""
#     category = categories[1]
#     days_range = [idx_range[0], idx_range[0]+5]
#     trend_range = [days_range[1],days_range[1]+5]
#     avg_range = [trend_range[1],idx_range[1]]
#     idx = data.index[data["Project"] == coin].tolist() #retrieve the row that belongs to the specific coin
#     data = data.iloc[idx,:]
#     data.reset_index(drop=True, inplace=True)
#     df_days = data.iloc[:,[x for x in range(days_range[0],days_range[1])]]
#     df_trend = data.iloc[:,[x for x in range(trend_range[0],trend_range[1])]]
#     df_range = data.iloc[:,[x for x in range(avg_range[0],avg_range[1])]]
#     df_days = df_days.transpose().reset_index()
#     df_days.rename(columns={"index":"Price_days", 0: "Value"},inplace=True)
#     df_trend = df_trend.transpose().reset_index()
#     df_trend.rename(columns={"index":"Price_trends", 0: "Value"},inplace=True)
#     df_range = df_range.transpose().reset_index()
#     df_range.rename(columns={"index":"Price_ranges", 0: "Value"},inplace=True)
#     fig1 = px.scatter(df_days, x=df_days["Price_days"], y=df_days["Value"])
#     fig2 = px.scatter(df_trend, x=df_trend["Price_trends"], y=df_trend["Value"])
#     fig3 = px.scatter(df_range, x=df_range["Price_ranges"], y=df_range["Value"])
#     return fig1,fig2,fig3
#
#
#
# def Annualized_plot(data,coin,idx_range):# 4,4,4
#     """Take in that dataframe and coin type and filter for only price related data and columns. there are days change and days trend change"""
#     category = categories[2]
#     revenues_range = [idx_range[0], idx_range[0] + 4]
#     days_range = [revenues_range[1], revenues_range[1]+4]
#     trend_range = [days_range[1],idx_range[1]]
#     idx = data.index[data["Project"] == coin].tolist() #retrieve the row that belongs to the specific coin
#     data = data.iloc[idx,:]
#     data.reset_index(drop=True, inplace=True)
#     df_revenue = data.iloc[:, [x for x in range(revenues_range[0], revenues_range[1])]]
#     df_days = data.iloc[:,[x for x in range(days_range[0],days_range[1])]]
#     df_trend = data.iloc[:,[x for x in range(trend_range[0],trend_range[1])]]
#     df_revenue = df_days.transpose().reset_index()
#     df_revenue.rename(columns={"index": "Revenue_days", 0: "Value"}, inplace=True)
#     df_days = df_days.transpose().reset_index()
#     df_days.rename(columns={"index":"Price_days" , 0: "Value"},inplace=True)
#     df_trend = df_trend.transpose().reset_index()
#     df_trend.rename(columns={"index":"Price_trends" , 0: "Value"},inplace=True)
#     fig1 = px.scatter(df_revenue, x=df_revenue["Revenue_days"], y=df_revenue["Value"])
#     fig2 = px.scatter(df_days, x=df_days["Price_days"], y=df_days["Value"])
#     fig3 = px.scatter(df_trend, x=df_trend["Price_trends"], y=df_trend["Value"])
#     return fig1,fig2,fig3
#
# def protocol_revenue_plot(data,coin,idx_range): #3,3,3,3,3
#     """Take in that dataframe and coin type and filter for only price related data and columns. there are days change and days trend change"""
#     category = categories[3:8]
#     range1 = [idx_range[0][0], idx_range[0][1]]
#     range2 = [idx_range[1][0], idx_range[1][1]]
#     range3 = [idx_range[2][0], idx_range[2][1]]
#     range4 = [idx_range[3][0], idx_range[3][1]]
#     range5 = [idx_range[4][0], idx_range[4][1]]
#     idx = data.index[data["Project"] == coin].tolist()  # retrieve the row that belongs to the specific coin
#     data = data.iloc[idx, :]
#     data.reset_index(drop=True, inplace=True)
#     df_range1 = data.iloc[:, [x for x in range(range1[0], range1[1])]]
#     df_range2 = data.iloc[:, [x for x in range(range2[0], range2[1])]]
#     df_range3 = data.iloc[:, [x for x in range(range3[0], range3[1])]]
#     df_range4 = data.iloc[:, [x for x in range(range4[0], range4[1])]]
#     df_range5 = data.iloc[:, [x for x in range(range5[0], range5[1])]]
#     df_range1 = df_range1.transpose().reset_index()
#     df_range1.rename(columns={"index": "Price_trends", 0: "Value"}, inplace=True)
#     df_range2 = df_range2.transpose().reset_index()
#     df_range2.rename(columns={"index": "Price_trends", 0: "Value"}, inplace=True)
#     df_range3 = df_range3.transpose().reset_index()
#     df_range3.rename(columns={"index": "Price_trends", 0: "Value"}, inplace=True)
#     df_range4 = df_range4.transpose().reset_index()
#     df_range4.rename(columns={"index": "Price_trends", 0: "Value"}, inplace=True)
#     df_range5 = df_range5.transpose().reset_index()
#     df_range5.rename(columns={"index": "Price_trends", 0: "Value"}, inplace=True)
#     fig1 = px.scatter(df_range1, x=df_range1["Price_trends"], y=df_range1["Value"])
#     fig2 = px.scatter(df_range2, x=df_range2["Price_trends"], y=df_range2["Value"])
#     fig3 = px.scatter(df_range3, x=df_range3["Price_trends"], y=df_range3["Value"])
#     fig4 = px.scatter(df_range4, x=df_range4["Price_trends"], y=df_range4["Value"])
#     fig5 = px.scatter(df_range5, x=df_range5["Price_trends"], y=df_range5["Value"])
#     return fig1,fig2,fig3,fig4,fig5
#
# def P_S_ratio_plot(data,coin,idx_range): #11,4
#     """Take in that dataframe and coin type and filter for only price related data and columns. there are days change and days trend change"""
#     category = categories[0]
#     ps_ratio_range = [idx_range[0], idx_range[0] + 10]
#     ps_ratio_day_range = [ps_ratio_range[1], idx_range[1]]
#     idx = data.index[data["Project"] == coin].tolist()  # retrieve the row that belongs to the specific coin
#     data = data.iloc[idx, :]
#     data.reset_index(drop=True, inplace=True)
#     df_ratio = data.iloc[:, [x for x in range(ps_ratio_range[0], ps_ratio_range[1])]]
#     df_avg_ratio = data.iloc[:, [x for x in range(ps_ratio_day_range[0], ps_ratio_day_range[1])]]
#     df_ratio = df_ratio.transpose().reset_index()
#     df_ratio.rename(columns={"index": "P/S ratio", 0: "Value"}, inplace=True)
#     df_avg_ratio = df_avg_ratio.transpose().reset_index()
#     df_avg_ratio.rename(columns={"index": "Avg P/S ratio", 0: "Value"}, inplace=True)
#     fig1 = px.scatter(df_ratio, x=df_ratio["P/S ratio"], y=df_ratio["Value"])
#     fig2 = px.scatter(df_avg_ratio, x=df_avg_ratio["Avg P/S ratio"], y=df_avg_ratio["Value"])
#     return fig1, fig2
#
# def P_E_ratio_plot(data,coin,idx_range):
#     """Take in that dataframe and coin type and filter for only price related data and columns. there are days change and days trend change"""
#     category = categories[0]
#     pe_ratio_range = [idx_range[0], idx_range[0] + 10]
#     pe_ratio_day_range = [pe_ratio_range[1], idx_range[1]]
#     idx = data.index[data["Project"] == coin].tolist()  # retrieve the row that belongs to the specific coin
#     data = data.iloc[idx, :]
#     data.reset_index(drop=True, inplace=True)
#     df_ratio = data.iloc[:, [x for x in range(pe_ratio_range[0], pe_ratio_range[1])]]
#     df_avg_ratio = data.iloc[:, [x for x in range(pe_ratio_day_range[0], pe_ratio_day_range[1])]]
#     df_ratio = df_ratio.transpose().reset_index()
#     df_ratio.rename(columns={"index": "P/E ratio", 0: "Value"}, inplace=True)
#     df_avg_ratio = df_avg_ratio.transpose().reset_index()
#     df_avg_ratio.rename(columns={"index": "Avg P/E ratio", 0: "Value"}, inplace=True)
#     fig1 = px.scatter(df_ratio, x=df_ratio["P/E ratio"], y=df_ratio["Value"])
#     fig2 = px.scatter(df_avg_ratio, x=df_avg_ratio["Avg P/E ratio"], y=df_avg_ratio["Value"])
#     return fig1, fig2
#
# print("price")
# #price_plot(df,"Cardano",columns_index[0])
# print("TVL")
# #TVL_plot(df,"Cardano",columns_index[1])
# print("Annualized")
# #Annualized_plot(df,"Cardano",columns_index[2])
# print("Protocol Revenue")
# protocol_revenue_plot(df,"Cardano",columns_index[3:8])
# print("P_S Ratio")
# P_S_ratio_plot(df,"Cardano",columns_index[8])
# print("P_E RAtio")
# P_E_ratio_plot(df,"Cardano",columns_index[8])
if __name__ == '__main__':
    app.run_server(debug=True, port=8056)

