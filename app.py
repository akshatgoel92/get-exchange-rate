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
from src import forex
from src import ui

# Set display format
pd.options.display.float_format = '{:,.2f}'.format

# Get summary and graph
sum, fig = forex.main()

# Load UI styles
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Set up application instance
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Graph
graph = dcc.Graph(figure=fig)

# Table
table = ui.gen_dash_table(sum)

# Create app layout
app.layout = html.Div([graph, table])


if __name__ == '__main__':
    app.run_server(debug=True)