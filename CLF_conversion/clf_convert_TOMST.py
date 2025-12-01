# PURPOSE: CONVERTS TOMST TMS4 AND TL DATA INTO A SHARED LOGGER FORMAT (CLF) AND EXPORTS FIGURES

import os
import pandas as pd
import geopandas as gpd
import plotly as plt 
import plotly.graph_objects as go
import numpy as np
import warnings

#####################
# SOURCE & SETTINGS #
#####################
source_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\1_CLF_conversion\source" # input path fll with .csv
output_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\1_CLF_conversion\output" # output path for CLF.csvs
figure_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\1_CLF_conversion\figures" # output path for figures
date_gate = "2024-03-01" # put in the start of your study date to reduce the amount of data the loop iterates over
#deviation_threshold = 0.6 # deviation threshold for the 2nd date gate which tries to find the start of the time series

#######################
# CLF CONVERSION LOOP #
#######################
for filename in os.listdir(source_folder):
    # COMMON PART 1 # 
    if filename.endswith('.csv') and 'data' in filename.lower(): # if its a csv file that contains the word data (TOMST standard)
        file_path = os.path.join(source_folder, filename)
        df = pd.read_csv(file_path, sep = ";", dayfirst = True, header = None, index_col=[0]) 
        #df[1] = pd.to_datetime(df[1], dayfirst=True, errors='coerce') # redundant datetime coercian
        df = df.rename(columns = {
            1: 'datetime',
            2: 'utc',
            3: 't1',
            4: 't2',
            5: 't3',
            6: 'SMC',
            7: 'shake',
            8: 'errFlag',
            9: 'bs'
        }) # rename columns
        df['datetime'] = pd.to_datetime(df['datetime'], format="%Y.%m.%d %H:%M") #datetime parser
        df = df[df['datetime'] >= pd.to_datetime(date_gate)] # apply first date gate
        df['datetime'] = df['datetime'].dt.tz_localize('UTC') # assign UTC timezone to datetime
        if (df['errFlag'] == 0).all(): # drop error flag column if 0 in every row
            df = df.drop(columns=['errFlag'])
        else:
                non_zero_count = (df['errFlag'] != 0).sum()
                warnings.warn(f"'errFlag' had {non_zero_count} non-zero rows. These rows were dropped.")
                df = df[df['errFlag'] == 0].copy()
                df = df.drop(columns=['errFlag'])

        # DETECT LOGGER TYPE AND MANIPULATE BASED ON DIFFERENCES
        if df[['t1', 't2', 'SMC']].isin([-200]).all().any():  # values in some columns are -200 = TL
            logger_type = 'TL'
            logcode = 2
            df = df.rename(columns={
                't1': 't3',
                't3': 't1',
            }) # tomst inconsistent in storage formatting
            # df['logtype'] = logcode # assign logger code maybe just unnecessary clutter since its in the name
            df[['t1', 't2', 'SMC']] = np.nan # fill CLF commons with nans
            df = df.reindex(['datetime', 't1', 't2', 't3', 'SMC'], axis=1)

        else: # else its a TMS
            logger_type = 'TMS'
            logcode = 1
            df = df.drop(columns=['utc', 'shake', 'bs'])
            # df['logtype'] = logcode
            columns_to_convert = ['t1', 't2', 't3']
            if df[columns_to_convert].applymap(type).eq(str).any().any():
                df[columns_to_convert] = df[columns_to_convert].apply(lambda x: x.str.replace(',', '.').astype(float))

        # COMMON PART 2 # 
        #dfsd = df.copy() # deviation filtering gate
        #dfsd['rolling_std_shrt'] = dfsd['t3'].rolling(window=3).std() # rolling std calculated over 3 days
        #dfsd['rolling_std_long'] = dfsd['rolling_std_shrt'].rolling(window=10).mean() # std deviation over 10 days
        #date_gate2 = dfsd[dfsd['rolling_std_long'] > deviation_threshold]['datetime'].min() # find first date in which the std deviation of 10 days exceeds the deviation threshold from the 3 days 
        #df = df[df['datetime'] >= pd.to_datetime(date_gate2)] # apply deviation filtering gate
        cleaned_filename = filename.replace("data", "").strip("_") 
        new_filename = f"CLF_{cleaned_filename[:-6]}_{logger_type}.csv"
        new_file_path = os.path.join(output_folder, new_filename)
        df.to_csv(new_file_path, index=False)
        print(f"{filename}: CLF format exported, proceed to Figure!")
        
        # FIGURES
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['t3'], mode='lines', name='T air'))
        if logger_type == 'TMS':  # Add more traces for TMS
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['t2'], mode='lines', name='T surface'))
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['t1'], mode='lines', name='T soil'))
        fig.update_layout(
            shapes=[dict(
                type="rect",
                xref="x", yref="paper",
                x0="2024-05-15", x1="2024-09-15",
                y0=0, y1=1,
                fillcolor="yellow", opacity=0.2, line_width=0,
            )],
            annotations=[dict(
                x='2024-07-15',
                y=1.05,
                xref='x',
                yref='paper',
                text='Summer 24',
                showarrow=False,
                font=dict(size=14, color="yellow"),
                align="center"
            )],
            title=f"{'Air temperature' if logger_type == 'TL' else 'Soil, surface, air temperature'}, {filename}",
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            hovermode='x unified',
            template='plotly_dark',
            height=700,
            width=1200
        )
        figure_name = f"CLF_{cleaned_filename[:-6]}_{logger_type}_fig.html"
        figure_path = os.path.join(figure_folder, figure_name)
        fig.write_html(figure_path)
        print(f"{filename}: Figure exported, proceed to next!")