import pandas as pd
from web3 import Web3
from utils import get_new_asset_data, merge_and_save, get_current_unix_timestamp, get_user_data, get_user_position_data
import logging
import streamlit as st


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

asset_data = pd.read_csv("./data/asset_data.csv")
user_data = pd.read_csv("./data/user_data.csv")
user_position_data = pd.read_csv("./data/user_position_data.csv")


current_unix_timestamp = get_current_unix_timestamp()

# Get new asset data if the timestamp is greater than 1 day
highest_timestamp_asset_data = asset_data["timestamp"].max()

if highest_timestamp_asset_data + 86400 < current_unix_timestamp:
    logger.info("Fetching new asset data")
    new_asset_data = get_new_asset_data()
    new_asset_data_df = pd.DataFrame(new_asset_data)
    merge_and_save(asset_data, new_asset_data_df, "./data/asset_data.csv")
    
else:
    logger.info("Using existing asset data")
    new_asset_data = asset_data.loc[asset_data['timestamp'].idxmax()].to_frame().transpose().to_dict(orient='records')[0]
    print(new_asset_data)
    new_asset_data_df = asset_data[asset_data['timestamp'] == highest_timestamp_asset_data]
    
# Get new user data if the timestamp is greater than 1 day
highest_timestamp_user_data = user_data["timestamp"].max() 

if highest_timestamp_user_data + 86400 < current_unix_timestamp:
    logger.info("Fetching new user data")
    new_user_data = get_user_data()
    merge_and_save(user_data, new_user_data, "./data/user_data.csv")
    
else:
    logger.info("Using existing user data")
    new_user_data = user_data[user_data['timestamp'] == highest_timestamp_user_data]
    

# Create a list of user addresses
user_addresses = new_user_data['user'].tolist()

users_checksum = []
for user in user_addresses:
    users_checksum.append(Web3.to_checksum_address(user))


highest_timestamp_user_position_data = user_position_data["timestamp"].max()

if highest_timestamp_user_position_data + 86400 < current_unix_timestamp:
    logger.info("Fetching new user position data")
    new_user_position_data = get_user_position_data(users_checksum, new_asset_data)
    merge_and_save(user_position_data, new_user_position_data, "./data/user_position_data.csv")

else:
    logger.info("Using existing user position data")
    new_user_position_data = user_position_data[user_position_data['timestamp'] == highest_timestamp_user_position_data]

logger.info("Data processing completed successfully")

extracted_asset_list = new_asset_data_df['symbol'].tolist()

# Create a dictionary mapping symbols to prices
price_dict = dict(zip(new_asset_data_df['symbol'], new_asset_data_df['price']))

# Function to get price for a column
def get_price(column_name):
    if column_name in ['user', 'timestamp']:
        return 1
    symbol = column_name[1:]  # Remove the 'a' or 'd' prefix
    return price_dict.get(symbol, 1)  # Default to 1 if symbol not found

# Create a new DataFrame with the same structure as user_position_data
new_df = new_user_position_data.copy()

# Add value columns for each position column
for column in new_user_position_data.columns:
    if column not in ['user', 'timestamp']:
        price = get_price(column)
        new_df[f'{column}_value'] = new_user_position_data[column] * price

logger.info("New DataFrame with value columns has been created")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd





# import plotly.graph_objects as go
# import pandas as pd
# import numpy as np
# import streamlit as st

# # Print column names to debug
# print("Columns in new_df:", new_df.columns.tolist())

# # Get all asset symbols (a{symbol})
# asset_symbols = [col[1:] for col in new_df.columns if col.startswith('a') and not col.endswith('_value')]

# # Get all debt symbols (d{symbol})
# debt_symbols = [col[1:] for col in new_df.columns if col.startswith('d') and not col.endswith('_value')]

# print("Asset symbols:", asset_symbols)
# print("Debt symbols:", debt_symbols)

# # Function to create heatmap for a single asset
# def create_heatmap(asset, data, sorted_debt_symbols):
#     fig = go.Figure(data=go.Heatmap(
#         z=[data],
#         x=sorted_debt_symbols,
#         y=[asset],
#         colorscale='viridis',
#         hoverongaps=False
#     ))
    
#     fig.update_layout(
#         title=f"Heatmap for {asset}",
#         xaxis_title="Debt Symbols",
#         yaxis_title="Asset",
#         height=300,  # Increased height
#         width=1000,  # Increased width
#         xaxis=dict(
#             tickangle=-90,  # Rotate x-axis labels by 90 degrees
#             side='bottom'
#         ),
#         yaxis=dict(tickangle=0),
#         annotations=[
#             dict(
#                 x=sorted_debt_symbols[i],
#                 y=asset,
#                 text=f'{val:.2e}',
#                 showarrow=False,
#                 font=dict(size=14),
#                 textangle=-90,  # Rotate text by 90 degrees
#                 yshift=65  # Move text above the heatmap
#             ) for i, val in enumerate(data)
#         ]
#     )
    
#     return fig

# # Create a heatmap for each asset
# for asset in asset_symbols:
#     heatmap_data = []
#     for debt in debt_symbols:
#         mask = new_df[f'a{asset}_value'] > 0
#         if f'd{debt}' in new_df.columns:
#             value = new_df.loc[mask, f'd{debt}_value'].sum()
#         else:
#             value = 0
#         if value >= 100:  # Only include values >= 100
#             heatmap_data.append((debt, value))
    
#     # If there's no data above 100, skip this asset
#     if not heatmap_data:
#         continue
    
#     # Sort the heatmap_data by value in descending order
#     heatmap_data.sort(key=lambda x: x[1], reverse=True)
    
#     # Separate the sorted data back into debt symbols and values
#     sorted_debt_symbols, sorted_values = zip(*heatmap_data)
    
#     fig = create_heatmap(asset, sorted_values, sorted_debt_symbols)
#     st.plotly_chart(fig, use_container_width=True)

# logger.info("Heatmaps have been created")

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

# Set the layout width to a wider size
st.set_page_config(layout="wide")

# Add title to your Streamlit app
st.title('Asset Debt Proportion Visualization')

# Print column names to debug
print("Columns in new_df:", new_df.columns.tolist())

# Get all asset symbols (a{symbol})
asset_symbols = [col[1:] for col in new_df.columns if col.startswith('a') and not col.endswith('_value')]

# Get all debt symbols (d{symbol})
debt_symbols = [col[1:] for col in new_df.columns if col.startswith('d') and not col.endswith('_value')]

print("Asset symbols:", asset_symbols)
print("Debt symbols:", debt_symbols)

# Function to create proportional bar chart for a single asset
def create_proportion_chart(asset, data, sorted_debt_symbols):
    total = sum(data)
    proportions = [val / total for val in data]
    
    fig = go.Figure(go.Bar(
        y=[asset] * len(data),
        x=data,
        orientation='h',
        marker=dict(
            color=proportions,
            colorscale='Viridis',
            showscale=True
        ),
        text=[f'{debt}: {val:.2e} ({prop:.2%})' for debt, val, prop in zip(sorted_debt_symbols, data, proportions)],
        textposition='inside',
        insidetextanchor='middle',
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"Debt Proportion for {asset}",
        xaxis_title="Value",
        yaxis_title="Asset",
        height=300,
        width=1000,
        bargap=0,
        xaxis=dict(
            type='linear',  # Use linear scale
            range=[0, max(data) * 1.1]  # Set range from 0 to slightly above the maximum value
        ),
        yaxis=dict(tickangle=0),
        coloraxis_showscale=True
    )
    
    return fig

# Create a dictionary to store the data for each asset
asset_data = {}

for asset in asset_symbols:
    heatmap_data = []
    for debt in debt_symbols:
        mask = new_df[f'a{asset}_value'] > 0
        if f'd{debt}' in new_df.columns:
            value = new_df.loc[mask, f'd{debt}_value'].sum()
        else:
            value = 0
        if value >= 100:  # Only include values >= 100
            heatmap_data.append((debt, value))
    
    # If there's no data above 100, skip this asset
    if not heatmap_data:
        continue
    
    # Sort the heatmap_data by value in descending order
    heatmap_data.sort(key=lambda x: x[1], reverse=True)
    
    # Separate the sorted data back into debt symbols and values
    sorted_debt_symbols, sorted_values = zip(*heatmap_data)
    
    asset_data[asset] = (sorted_debt_symbols, sorted_values)

# Create a dropdown for asset selection
selected_asset = st.selectbox('Select an asset:', list(asset_data.keys()))

# Display the chart for the selected asset
if selected_asset in asset_data:
    sorted_debt_symbols, sorted_values = asset_data[selected_asset]
    fig = create_proportion_chart(selected_asset, sorted_values, sorted_debt_symbols)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data available for the selected asset.")

logger.info("Proportion chart has been created for the selected asset")