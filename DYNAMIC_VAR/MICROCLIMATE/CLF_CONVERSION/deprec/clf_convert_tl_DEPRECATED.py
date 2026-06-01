import os
import pandas as pd
import geopandas as gpd
import plotly as plt 
import plotly.graph_objects as go
import numpy as np

#####################
# SOURCE & SETTINGS #
#####################
source_folder = r"\\ad.helsinki.fi\home\t\terschan\Documents\csv_test\TL\source" # input path fll with .csv
output_folder = r"\\ad.helsinki.fi\home\t\terschan\Documents\csv_test\TL\output" # output path for CLF.csvs
figure_folder = r"\\ad.helsinki.fi\home\t\terschan\Documents\csv_test\TL\figures" # output path for figures
date_gate = "2024-03-01" # rough date at which data becomes available, here: start of the study
logtype = 2 # 1 = TMS, 2 = TL, 3 = ST, 4 = NA
deviation_threshold = 0.6

#######################
# CLF CONVERSION LOOP #
#######################
# 1) converts Thermologger data format in CLF, 2) creates interactive .html figures of the data and dumps them into a folder
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        # IMPORT FILE
        file_path = os.path.join(source_folder, filename)
        df = pd.read_csv(file_path, sep = ";", parse_dates = [1], dayfirst = True, header = None, index_col=[0]) 
        # MANIPULATE FILE
        df = df.rename(columns = {
            1: 'datetime',
            2: 'utc',
            3: 't3',
            4: 't2',
            5: 't1',
            6: 'SMC',
            7: 'shake',
            8: 'errFlag',
            9: 'bs'
        })
        df = df[df['datetime'] >= pd.to_datetime(date_gate)]
        # DEVIATION FILTERING GATE APPROACH, THIS NEEDS SOME FINE TUNING AND OPTIMIZING 
        dfsd = df.copy()
        dfsd['rolling_std_shrt'] = dfsd['t3'].rolling(window=3).std()
        dfsd['rolling_std_long'] = dfsd['rolling_std_shrt'].rolling(window=14).mean()
        date_gate2 = dfsd[dfsd['rolling_std_long'] > deviation_threshold]['datetime'].min()
        df = df[df['datetime'] >= pd.to_datetime(date_gate2)]
        # end of deviation filter gate
        df['datetime'] = df['datetime'].dt.tz_localize('UTC') 
        # drop error flag column if = 0
        if (df['errFlag'] == 0).all():
            df = df.drop(columns=['errFlag']) 
        else:
            raise ValueError("Error Flag does not consist entirely of zeros and cannot be removed.")
        df = df.drop(columns=['utc', 'shake', 'bs']) # drop unnecessary columns
        # add logtype column CHANGE
        df['logtype'] = logtype 
        # fill NA colums with NAN
        columns_to_replace = ['t1', 't2', 'SMC']
        df[columns_to_replace] = np.nan
        # switch order of columns to comply with CLF 
        df = df.reindex(['datetime','t1','t2', 't3', 'SMC', 'logtype'], axis=1)
        # EXPORT FILE
        new_filename = f"{logtype}_CLF_{filename[:-6]}.csv"  # Adding CLF prefix (common logger format) and logger suffix
        new_file_path = os.path.join(output_folder, new_filename)
        df.to_csv(new_file_path, index=False)
        print(f"{filename}: CLF format exported, proceed to Figure!")
        # PART 2: FIGURES
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['t3'], 
                         mode='lines', name='T air'))
        fig.update_layout(
            shapes=[dict( # Summer period
                        type="rect",
                        xref="x", yref="paper",
                        x0="2024-05-15", x1="2024-09-15",
                        y0=0, y1=1,
                        fillcolor="yellow", opacity=0.2, line_width=0,
                    )])
        fig.update_layout(
            annotations=[dict(
                    x='2024-07-15',  # Position the text in the middle of the summer region
                    y=1.05,  # Position the text above the plot
                    xref='x',
                    yref='paper',
                    text='Summer 24',
                    showarrow=False,
                    font=dict(size=14, color="yellow"),  # Set font size and color
                    align="center"
                    )])
        fig.update_layout(title=f'Air temperature, {filename}',
                  xaxis_title='Date',
                  yaxis_title='Temperature (°C)',
                  hovermode='x unified',  # show hover info for both lines on the same x-value
                  template='plotly_dark',
                  height=700,   # Adjust this to make the plot taller
                  width=1200 )  # You can change this to 'plotly_white' if preferred
        # EXPORT HTML FIGURES
        figure_name = f"{logtype}_CLF_{filename[:-6]}_figure.html"  # Adding CLF prefix (common logger format) and logger suffix
        figure_path = os.path.join(figure_folder, figure_name)
        fig.write_html(figure_path)
        print(f"{filename}: Figure exported, proceed to next!")