"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.subplots as subplots
import pandas
from dotenv import dotenv_values
import plotly.io as pio

pio.renderers.default = "browser"
seeds = [1, 2, 3]
folds = [0, 1, 2]
splits = ['1day', '1week', '2weeks', '3weeks', '4weeks']

# Load variables in .env file
config = dotenv_values("../.env")
data_path = os.path.join(config['data_path'], 'dsq')
model_path = os.path.join(config['model_path'], 'dsq')

time_axis = [1, 7, 14, 21, 28]

fig = subplots.make_subplots(rows=3, cols=1, subplot_titles=('B=1', 'B=5', 'B=10'), vertical_spacing=0.07)
for results_idx in range(3):
    df_upper_baseline = pandas.read_excel(
        os.path.join(model_path, f'results.xlsx'),
        header=0,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7],
        sheet_name='best'
    )

    df_lower_baseline = pandas.read_excel(
        os.path.join(model_path, f'results.xlsx'),
        header=0,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7],
        sheet_name='unsupervised'
    )
    df_upper_baseline_mean = df_upper_baseline.groupby('Split')['F1'].mean()
    df_lower_baseline_mean = df_lower_baseline.groupby('Split')['F1'].mean()

    # Add a helper trace for the y=1 line
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=[0.7] * len(time_axis),  # A constant y=1 for all x values
        line=dict(color='rgba(0, 0, 0, 0)'),  # Invisible line
        showlegend=False
    ), row=results_idx + 1, col=1)

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=df_upper_baseline_mean,
        line=dict(color='gray'),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 0, 0, 0.2)',  # Adjust the alpha value for transparency
    ), row=results_idx + 1, col=1)

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=df_lower_baseline_mean,
        line=dict(color='gray'),
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0, 0, 0, 0.2)',  # Adjust the alpha value for transparency
    ), row=results_idx + 1, col=1)

    approach_list = ['rand', 'unc', 'top', 'ds']
    approach_name_list = ['random', 'uncertainty', 'top', 'disimilarity']
    colour_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    sheet_names = pandas.ExcelFile(os.path.join(model_path, 'results.xlsx')).sheet_names
    for approach_idx, approach in enumerate(approach_list):
        combination = []
        for sheet in sheet_names:
            if approach in sheet:
                combination.append(sheet)

        df = pandas.read_excel(
            os.path.join(model_path, f'results.xlsx'),
            sheet_name=combination[results_idx],
            header=0,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7],
            nrows=45
        )

        df_mean = df.groupby('Split')['F1'].mean()
        df_std = df.groupby('Split')['F1'].std()

        fig.add_trace(go.Scatter(
            x=time_axis,
            y=df_mean,
            line=dict(color=colour_list[approach_idx]),
            name=approach_name_list[approach_idx],
        ), row=results_idx + 1, col=1)

    if results_idx == 2:
        x_axis_label = 'Time [days]'
    else:
        x_axis_label = ''

    fig.update_xaxes(
        linecolor='black',
        showgrid=False,
        mirror=True,
        range=[1, 28],
        title_text=x_axis_label,
        row=results_idx + 1, col=1
    )

    fig.update_yaxes(
        linecolor='black',
        showgrid=True,
        gridcolor='black',
        mirror=True,
        range=[0, 0.7],
        title_text='F_1',
        row=results_idx + 1, col=1
    )

    fig.update_layout(
        showlegend=True,
        plot_bgcolor="white",
        font=dict(size=20, family="Times New Roman", color='black'),
        height=1500,
        width=1500,
        legend=dict(
            x=0.4,  # Horizontal position (0=left, 1=right)
            y=0.8,  # Vertical position (0=bottom, 1=top)
            bgcolor='rgba(255, 255, 255, 1)',  # Background color with transparency
        )
    )

for i in fig['layout']['annotations']:
    i['font'] = dict(size=20, family="Times New Roman", color='black')

fig.show()
fig.write_image(r"/home/lcs_crr/Downloads/budget.svg")

fig = subplots.make_subplots(rows=3, cols=1, subplot_titles=('p_m=0.1', 'p_m=0.2', 'p_m=0.3'), vertical_spacing=0.07)
for results_idx in range(3, 6):
    df_upper_baseline = pandas.read_excel(
        os.path.join(model_path, f'results.xlsx'),
        header=0,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7],
        sheet_name='best'
    )

    df_lower_baseline = pandas.read_excel(
        os.path.join(model_path, f'results.xlsx'),
        header=0,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7],
        sheet_name='unsupervised'
    )
    df_upper_baseline_mean = df_upper_baseline.groupby('Split')['F1'].mean()
    df_lower_baseline_mean = df_lower_baseline.groupby('Split')['F1'].mean()

    # Add a helper trace for the y=1 line
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=[0.7] * len(time_axis),  # A constant y=1 for all x values
        line=dict(color='rgba(0, 0, 0, 0)'),  # Invisible line
        showlegend=False
    ), row=results_idx - 2, col=1)

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=df_upper_baseline_mean,
        line=dict(color='gray'),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 0, 0, 0.2)',  # Adjust the alpha value for transparency
    ), row=results_idx - 2, col=1)

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=df_lower_baseline_mean,
        line=dict(color='gray'),
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0, 0, 0, 0.2)',  # Adjust the alpha value for transparency
    ), row=results_idx - 2, col=1)

    approach_list = ['rand', 'unc', 'top', 'ds']
    approach_name_list = ['random', 'uncertainty', 'top', 'disimilarity']
    colour_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for approach_idx, approach in enumerate(approach_list):
        combination = []
        for sheet in sheet_names:
            if approach in sheet:
                combination.append(sheet)

        df = pandas.read_excel(
            os.path.join(model_path, f'results.xlsx'),
            sheet_name=combination[results_idx],
            header=0,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7],
            nrows=45
        )

        df_mean = df.groupby('Split')['F1'].mean()
        df_std = df.groupby('Split')['F1'].std()

        fig.add_trace(go.Scatter(
            x=time_axis,
            y=df_mean,
            line=dict(color=colour_list[approach_idx]),
            name=approach_name_list[approach_idx],
        ), row=results_idx - 2, col=1)

    if results_idx == 2 or results_idx == 5:
        x_axis_label = 'Time [days]'
    else:
        x_axis_label = ''

    fig.update_xaxes(
        linecolor='black',
        showgrid=False,
        mirror=True,
        range=[1, 28],
        title_text=x_axis_label,
        row=results_idx - 2, col=1
    )

    fig.update_yaxes(
        linecolor='black',
        showgrid=True,
        gridcolor='black',
        mirror=True,
        range=[0, 0.7],
        title_text='F_1',
        row=results_idx - 2, col=1
    )

    fig.update_layout(
        showlegend=True,
        plot_bgcolor="white",
        font=dict(size=20, family="Times New Roman", color='black'),
        height=1500,
        width=1500,
        legend=dict(
            x=0.4,  # Horizontal position (0=left, 1=right)
            y=0.8,  # Vertical position (0=bottom, 1=top)
            bgcolor='rgba(255, 255, 255, 1)',  # Background color with transparency
        )
    )

for i in fig['layout']['annotations']:
    i['font'] = dict(size=20, family="Times New Roman", color='black')

fig.show()
fig.write_image(r"/home/lcs_crr/Downloads/mislabeling.svg")
