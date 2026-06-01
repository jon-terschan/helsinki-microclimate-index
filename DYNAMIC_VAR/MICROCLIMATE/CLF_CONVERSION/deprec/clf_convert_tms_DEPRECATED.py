import os
import pandas as pd
import geopandas as gpd
import plotly as plt 
import plotly.graph_objects as go

#####################
# SOURCE & SETTINGS #
#####################
source_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\clf_conversion\source" # input path fll with .csv
output_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\clf_conversion\output" # output path for CLF.csvs
figure_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\clf_conversion\figures" # output path for figures
cutoff_date = "2024-03-01" # rough date at which data becomes available, here: start of the study
logtype = 1 # 1 = TMS, 2 = TL, 3 = ST, 4 = NA

#######################
# CLF CONVERSION LOOP #
#######################
# 1) converts TMS-4 data format in CLF, 2) creates interactive .html figures of the data and dumps them into a folder
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        # IMPORT FILE
        file_path = os.path.join(source_folder, filename)
        df = pd.read_csv(file_path, sep = ";", dayfirst = True, header = None, index_col=[0]) 
        # MANIPULATE FILE
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
        })
        df['datetime'] = pd.to_datetime(df['datetime'], format="%Y.%m.%d %H:%M")
        df = df[df['datetime'] >= pd.to_datetime(cutoff_date)]
        df['datetime'] = df['datetime'].dt.tz_localize('UTC') 
        # drop error flag column if = 0
        if (df['errFlag'] == 0).all():
            df = df.drop(columns=['errFlag']) 
        else:
            raise ValueError("Error Flag does not consist entirely of zeros and cannot be removed.")
        df = df.drop(columns=['utc', 'shake', 'bs']) # drop unnecessary columns
        # add logtype column CHANGE
        logtype = 1 # 1 = TMS, 2 = TL, 3 = ST, 4 = NA
        df['logtype'] = logtype 
        # convert temperature columns to float and replace . with ,
        columns_to_convert = ['t1', 't2', 't3'] 
        for col in columns_to_convert:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        # EXPORT FILE
        new_filename = f"{logtype}_CLF_{filename[:-6]}.csv"  # Adding CLF prefix (common logger format) and logger suffix
        new_file_path = os.path.join(output_folder, new_filename)
        df.to_csv(new_file_path, index=False)
        print(f"{filename}: CLF format exported, proceed to Figure!")
        # PART 2: FIGURES
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['t3'], 
                         mode='lines', name='T air'))
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['t2'], 
                         mode='lines', name='T surface'))
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['t1'], 
                         mode='lines', name='T soil'))
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
        fig.update_layout(title=f'Soil, surface, air temperature, {filename}',
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

# GENERAL IDEA OF THE DEVIATION THRESHOLD IDENTIFIER
#date_df = temp_df # put into second df
#date_df.set_index('datetime', inplace=True) # set datetime as index
#date_df['deviation_from_21'] = temp_df['t1'] - 21
#date_df = 6 # deviation threshold
#date_df['sustained_deviation'] = abs(temp_df['deviation_from_21']) >= threshold # binary mask
#transition_date = date_df[date_df['sustained_deviation']].index[0] # find first occurence of sustained deviation
#print(f"The approximate date the logger was placed in the field is: {transition_date}")

#dfsd = df.copy()
#dfsd['rolling_std'] = dfsd['t3'].rolling(window=3).std()
#Q1 = dfsd['t3'].quantile(0.25)
#Q3 = dfsd['t3'].quantile(0.75)
#IQR = Q3 - Q1
#lower_bound = Q1 - 2 * IQR
#upper_bound = Q3 + 2 * IQR
#df_filtered = dfsd[(dfsd['t3'] >= lower_bound) & (dfsd['t3'] <= upper_bound)]
#threshold = df_filtered['rolling_std'].mean() + 2 * df_filtered['rolling_std'].std()
#cutoff_date = df_filtered[df_filtered['rolling_std'] > threshold]['datetime'].min()
#df = df[df['datetime'] >= pd.to_datetime(cutoff_date)]