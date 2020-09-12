# Import packages
import json
import dash
import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Dash dependencies
import dash_core_components as dcc
import dash_html_components as html

# From dependencies
from dash.dependencies import Input, Output


def get_plot(df):
    '''
    ------------------
    Add moving avgs
    ------------------
    '''
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df.INR,
                             mode='lines',
                             name='Spot Rate',
                             line=dict(color='Black', width=1, dash='longdash')))
    
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df.sma,
                             mode='lines',
                             name='Short-term avg.'))
    
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df.lma,
                             mode='lines', 
                             name='Long-term avg.'))

    fig.update_layout(title='GBP INR Over Time',
                      xaxis_title='Date',
                      yaxis_title='INR', 
                      height = 575, 
                      annotations = [dict(
                                        xref='paper',
                                        yref='paper',
                                        y=-0.17,
                                        x=0,
                                        showarrow=False,
                                        text ="""Data Source: European Central Bank."""), 
                                     dict(
                                        xref='paper',
                                        yref='paper',
                                        y=-0.199,
                                        x=0,
                                        showarrow=False,
                                        text ='Buy GBP if red crosses green from below. Sell GBP if red crosses green from above.')])


    return(fig)


def get_dash_table(df):
    '''
    ------------------
    Add moving avgs
    ------------------
    '''
    table = dash_table.DataTable(id='table', 
                                 columns=[{"name": i.upper(), "id": i} 
                                          for i in df.columns],
                                 data=df.round(2).to_dict('records'), 
                                 style_header={ 'border': '1px solid black'},
                                 style_cell={ 'border': '1px solid grey', 
                                              'minWidth': '10%', 
                                              'width': '10%', 
                                              'maxWidth': '10%',
                                              'textAlign': 'left'}, 
                                 style_table={ 'width': '20%', 'margin': '7%'},)

    return(table)
