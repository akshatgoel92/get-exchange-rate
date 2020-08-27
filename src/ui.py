# Import packages
import json
import dash
import dash_table
import pandas as pd
import plotly.express as px

# Dash dependencies
import dash_core_components as dcc
import dash_html_components as html

# From dependencies
from dash.dependencies import Input, Output

def generate_table(df, max_rows=10):
    
    return(html.Table([html.Thead(html.Tr([html.Th(col) for col in df.columns])),
           html.Tbody([html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) 
           for i in range(min(len(df), max_rows))])]))
