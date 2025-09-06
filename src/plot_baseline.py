"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""
import os

import numpy as np
import plotly
import plotly.graph_objects as go
import pandas
from dotenv import dotenv_values

seeds = [1, 2, 3, 4, 5]
folds = [0, 1, 2]
splits = ['1day', '1week', '2weeks', '3weeks', '4weeks']

# Declare constants
MODEL_NAME = 'tevae'
BUDGET = 10
MISLABEL_PROB = 0
REVERSE_WINDOW_MODE = 'mean'
WINDOW_SIZE = 256
SAMPLING_RATE = 2

# Load variables in .env file
config = dotenv_values("../.env")
data_path = config['data_path']
model_path = config['model_path']

time_axis = [1, 7, 14, 21, 28]

fig = go.Figure()

df_upper_baseline = pandas.read_excel(os.path.join(model_path, f'results_best.xlsx'),
                                      # sheet_name=combination,
                                      header=0,
                                      usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                      )

df_lower_baseline = pandas.read_excel(os.path.join(model_path, f'results_baseline.xlsx'),
                                      # sheet_name=combination,
                                      header=0,
                                      usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                      )
df_upper_baseline_mean = df_upper_baseline.groupby('Split').mean().reset_index().drop(columns=['Split', 'Fold', 'Seed'])
df_lower_baseline_mean = df_lower_baseline.groupby('Split').mean().reset_index().drop(columns=['Split', 'Fold', 'Seed'])

colour_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig.add_trace(go.Scatter(
    x=time_axis,
    y=df_upper_baseline_mean['F1'],
    line=dict(color=colour_list[0]),
    mode='lines',
    # fill='tonexty',
    # fillcolor='rgba(0, 0, 0, 0.2)',  # Adjust the alpha value for transparency
))

fig.add_trace(go.Scatter(
    x=time_axis,
    y=df_lower_baseline_mean['F1'],
    line=dict(color=colour_list[1]),
    mode='lines',
    # fill='tozeroy',
    # fillcolor='rgba(0, 0, 0, 0.2)',  # Adjust the alpha value for transparency
))

fig.update_xaxes(
    linecolor='black',
    showgrid=False,
    # gridcolor='gray',
    # gridwidth=0.3,
    # showticklabels=False,
    mirror=True,
    range=[1, 28],
    title_text='Time (days)',
)

fig.update_yaxes(
    linecolor='black',
    showgrid=True,
    gridcolor='black',
    # gridwidth=0.3,
    # showticklabels=False,
    mirror=True,
    # range=[0, 0.8],
    range=[0, 1],
    title_text='F<sub>1</sub>',
)
fig.update_layout(
    showlegend=False,
    plot_bgcolor="white",
    font=dict(size=20, family="Times New Roman", color='black'),
    height=500,
    width=1500,
)
# fig.write_html(os.path.join(model_path, 'plots', combination + '.html'))
import plotly.io as pio

pio.renderers.default = "browser"
fig.show()
print()
