# Import packages
import json
import dash
import torch
import dash_table
import pandas as pd
import plotly.express as px

# Dash dependencies
import dash_core_components as dcc
import dash_html_components as html

# From dependencies
from dash.dependencies import Input, Output
from models import gp
from src import fetch
from src import ui


def main():
    
    # Get summary and graph
    sum, df, train_x, train_y = fetch.main()

    # Get machine learning predictions
    gp.main(train_x, train_y)


if __name__ == '__main__':
    main()