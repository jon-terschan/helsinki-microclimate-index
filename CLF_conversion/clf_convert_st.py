import os
import pandas as pd
import geopandas as gpd
import plotly as plt 
import plotly.graph_objects as go
import numpy as np
import pytz
import warnings

#####################
# SOURCE & SETTINGS #
#####################
source_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\clf_conversion\source" # input path fll with .csv
output_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\clf_conversion\output" # output path for CLF.csvs
figure_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\clf_conversion\figures" # output path for figures
# cutoff_date = "2024-03-01" # unnecessary because ST logs only when setup
logtype = 3 # 1 = TMS, 2 = TL, 3 = ST, 4 = NA
logger_type = "ST"
local_tz = pytz.timezone('Europe/Helsinki') # hand initial timezone for UTC conversion, ST record in local time of the computer they have been set up to
#######################
# CLF CONVERSION LOOP #
#######################
# DESC: 1) converts ST data format in CLF, 2) creates interactive .html figures of the data and dumps them into a figures folder, open issues = UTC

for filename in os.listdir(source_folder):
    if filename.endswith('.csv') and 'SurveyTag' in filename:
        # IMPORT FILE
        file_path = os.path.join(source_folder, filename)
        df = pd.read_csv(file_path, sep = ",", header = 0, skiprows = 3)  
        # MANIPULATE FILE
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']]) # create datetime column
        df['datetime'] = df['datetime'].apply(lambda x: local_tz.localize(x)) # localize to local timezone
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')  # convert to UTC
        df = df.drop(columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'CJ_Reading', 'vBatt', 'Samples']) #drop unnecessary columns
        if (df['Fault_Code'] == 0).all(): # drop fault code column if 0, raise error if not
            df = df.drop(columns=['Fault_Code']) 
        else:
            non_zero_count = (df['Fault_Code'] != 0).sum()
            warnings.warn(f"'errFlag' had {non_zero_count} non-zero rows. These rows were dropped.")
            df = df[df['Fault_Code'] == 0].copy()
            df = df.drop(columns=['Fault_Code'])   
        df= df.rename(columns = { # rename temp readings
                'TC_Reading':'t3'  }
                )
        #df['logtype'] = logtype # kinda unnecessary, could be in name
        columns_to_replace = ['t1', 't2', 'SMC'] # empty columns for other measurements
        df[columns_to_replace] = np.nan
        df = df.reindex(['datetime','t1','t2', 't3', 'SMC', 
                         #'logtype'
                         ], axis=1) # rearrange column order to CLF
        # EXPORT FILE
        new_filename = f"CLF_{filename[:-4]}_{logger_type}.csv"  # Adding CLF prefix (common logger format) and logger suffix
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
        figure_name = f"CLF_{filename[:-4]}_{logger_type}_fig.html"  # Adding CLF prefix (common logger format) and logger suffix
        figure_path = os.path.join(figure_folder, figure_name)
        fig.write_html(figure_path)
        print(f"{filename}: Figure exported, proceed to next!")

###########################
# Battery Drainage Report #
###########################
# DESC: Creates .csv report on the difference in battery power recorded by the ST loggers within the time series, exports to 'output_folder'

battery = [] # empty list to append to

for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(source_folder, filename)
        df = pd.read_csv(file_path, sep = ",", header = 0, skiprows = 3)  
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']]) # create datetime column for export
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        difference = first_row['vBatt'] - last_row['vBatt']
        battery.append({
                "logger" : filename,
                "start_date": first_row['datetime'],
                "end_date": last_row['datetime'],
                "difference": difference,
        }) # append to empty list

# turn filled list into df and export to output folder
batterydf = pd.DataFrame(battery)
battery_path = os.path.join(output_folder, "battery_drainage_report.csv")
batterydf.to_csv(battery_path, index=False)

# GENERAL IDEA OF THE DEVIATION THRESHOLD IDENTIFIER
#date_df = temp_df # put into second df
#date_df.set_index('datetime', inplace=True) # set datetime as index
#date_df['deviation_from_21'] = temp_df['t1'] - 21
#date_df = 6 # deviation threshold
#date_df['sustained_deviation'] = abs(temp_df['deviation_from_21']) >= threshold # binary mask
#transition_date = date_df[date_df['sustained_deviation']].index[0] # find first occurence of sustained deviation
#print(f"The approximate date the logger was placed in the field is: {transition_date}")