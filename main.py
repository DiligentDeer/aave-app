import pandas as pd
from web3 import Web3
from utils import get_new_asset_data, merge_and_save, get_current_unix_timestamp, get_user_data, get_user_position_data
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
import locale
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import numpy as np

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

# Set the layout width to a wider size
st.set_page_config(layout="wide")

# Add title to your Streamlit app
st.title('Asset Debt Proportion Visualization')


# Get all asset symbols (a{symbol})
collateral_symbols = [col[1:] for col in new_df.columns if col.startswith('a') and not col.endswith('_value')]

# Get all debt symbols (d{symbol})
debt_symbols = [col[1:] for col in new_df.columns if col.startswith('d') and not col.endswith('_value')]

# Set the locale for number formatting
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def create_proportion_charts(asset, data, sorted_debt_symbols, new_asset_data):
    total = sum(data)
    threshold = 0.01  # 1% threshold
    
    # Convert new_asset_data to DataFrame if it's a list
    if isinstance(new_asset_data, list):
        new_asset_data = pd.DataFrame(new_asset_data)
    
    # Create a list of tuples (debt, value, proportion)
    debt_data = list(zip(sorted_debt_symbols, data, [val / total for val in data]))
    
    # Separate data into main categories and others
    main_categories = [item for item in debt_data if item[2] >= threshold]
    others = [item for item in debt_data if item[2] < threshold]
    
    # Add "Others" category for the pie chart only
    if others:
        others_value = sum(item[1] for item in others)
        others_proportion = sum(item[2] for item in others)
        main_categories_with_others = main_categories + [("Others", others_value, others_proportion)]
    else:
        main_categories_with_others = main_categories
    
    # Sort main categories by value (descending order)
    main_categories_with_others.sort(key=lambda x: x[1], reverse=True)
    
    # Unzip the sorted data for the pie chart
    labels_pie, values_pie, proportions_pie = zip(*main_categories_with_others)
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'xy'}]])
    
    # Add pie chart
    fig.add_trace(go.Pie(
        labels=labels_pie,
        values=values_pie,
        textinfo='percent',
        hoverinfo='label+value+percent',
        marker=dict(colors=px.colors.qualitative.Set3)
    ), 1, 1)
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=[label for label in labels_pie if label != "Others"],
        y=[prop * 100 for label, prop in zip(labels_pie, proportions_pie) if label != "Others"],
        text=[f'{prop:.1%}' for label, prop in zip(labels_pie, proportions_pie) if label != "Others"],
        textposition='auto',
        marker_color=px.colors.qualitative.Set3[:len(labels_pie)-1]
    ), 1, 2)
    
    # Update layout
    fig.update_layout(
        title=f"Debt Proportion for {asset}",
        height=500,
        width=1200,  # Increased width to accommodate both charts
    )
    
    # Update bar chart axis
    fig.update_yaxes(title_text='Proportion (%)', row=1, col=2)
    fig.update_xaxes(title_text='Debt Assets', row=1, col=2)
    
    
    
    
    
    # Create table data (excluding "Others")
    table_data = pd.DataFrame({
        'Debt': [item[0] for item in main_categories],
        'Value': [item[1] for item in main_categories],
        'Proportion': [f'{item[2]:.2%}' for item in main_categories]
    })
    
    # Add new columns
    for label in table_data['Debt']:
        asset_data = new_asset_data[new_asset_data['symbol'] == label]
        if not asset_data.empty:
            asset_data = asset_data.iloc[0]
            table_data.loc[table_data['Debt'] == label, 'Borrow Cap'] = asset_data['borrowCap']
            table_data.loc[table_data['Debt'] == label, '% of Borrow Cap'] = 100 * asset_data['debtSupply'] / asset_data['borrowCap'] if asset_data['borrowCap'] != 0 else 0
            table_data.loc[table_data['Debt'] == label, 'Current Borrow'] = asset_data['debtSupply']
            table_data.loc[table_data['Debt'] == label, 'Current Borrow $'] = asset_data['debtSupply'] * asset_data['price']
            
    
    # Calculate % of Current Borrow
    table_data['% of Current Borrow'] = table_data['Value'] / table_data['Current Borrow $'] * 100
    
    return fig, table_data

# Create a dictionary to store the data for each asset
collateral_data = {}

for asset in collateral_symbols:
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
    
    collateral_data[asset] = (sorted_debt_symbols, sorted_values)


# New section: Create a dictionary to store the data for each debt
debt_data = {}

for debt in debt_symbols:
    heatmap_data = []
    for asset in collateral_symbols:
        mask = new_df[f'd{debt}_value'] > 0
        if f'a{asset}' in new_df.columns:
            value = new_df.loc[mask, f'a{asset}_value'].sum()
        else:
            value = 0
        if value >= 100:  # Only include values >= 100
            heatmap_data.append((asset, value))
    
    # If there's no data above 100, skip this debt
    if not heatmap_data:
        continue
    
    # Sort the heatmap_data by value in descending order
    heatmap_data.sort(key=lambda x: x[1], reverse=True)
    
    # Separate the sorted data back into collateral symbols and values
    sorted_collateral_symbols, sorted_values = zip(*heatmap_data)
    
    debt_data[debt] = (sorted_collateral_symbols, sorted_values)

def format_value(val, column):
        if pd.isna(val) or val == '':
            return 'N/A'
        try:
            float_val = float(val)
            if column in ['Value', 'Current Borrow $']:
                return f'${float_val:,.0f}'
                # return f'${locale.format_string("%,.2f", float_val, grouping=True)}'
            elif column in ['% of Borrow Cap', '% of Current Borrow']:
                return f'{float_val:.2f}%'
            else:
                return f'{float_val:,.0f}'
        except (ValueError, TypeError):
            return str(val)
        
def format_table_data(table_data):
    formatted_table_data = table_data.copy()
    for col in formatted_table_data.columns:
        if col not in ['Debt', 'Proportion']:  # Skip non-numeric columns
            formatted_table_data[col] = formatted_table_data[col].apply(lambda x: format_value(x, col))
    return formatted_table_data
        
        
        
# Create a dropdown for asset selection
selected_asset = st.selectbox('Select an asset:', list(collateral_data.keys()))

# Display detailed information for the selected asset
if selected_asset in collateral_data:
    st.header(f"Detailed Information for {selected_asset}")
    
    # Get the asset data
    asset_info = new_asset_data_df[new_asset_data_df['symbol'] == selected_asset].iloc[0]
    
    collateral_supply_value = asset_info['collateralSupply'] * asset_info['price']
    debt_supply_value = asset_info['debtSupply'] * asset_info['price']
    utilization_rate = debt_supply_value / collateral_supply_value if collateral_supply_value > 0 else 0
    
    borrow_cap_ratio = asset_info['debtSupply'] / asset_info['borrowCap']
    supply_cap_ratio = asset_info['collateralSupply'] / asset_info['supplyCap']
    
    # Create three columns for layout
    col1, col2, col3, col4 = st.columns(4)
    
    
    with col1:
        st.metric("Current Price", f"${asset_info['price']:,.2f}")
        st.metric("LTV", f"{asset_info['ltv']:.2%}")
        st.metric("Liquidation Threshold", f"{asset_info['liquidationThreshold']:.2%}")
    
    with col2:
        st.metric("Borrow Cap", f"{asset_info['borrowCap']:,.0f}")
        st.metric("Supply Cap", f"{asset_info['supplyCap']:,.0f}")
        st.metric("Liquidation Bonus", f"{asset_info['liquidationBonus']:.2%}")
    
    with col3:
        st.metric("Current Debt", f"{asset_info['debtSupply']:,.0f}")
        st.metric("Current Supply", f"{asset_info['collateralSupply']:,.0f}")
        st.metric("Reserve Factor", f"{asset_info['reserveFactor']:.2%}")
    
    with col4:
        st.metric("% Lent", f"{borrow_cap_ratio:,.2%}")
        st.metric("% Supplied", f"{supply_cap_ratio:.2%}")
        st.metric("Utilization Rate", f"{utilization_rate:.2%}")
        
    
    
    

    # Display asset addresses
    st.subheader("Asset Addresses")
    st.write(f"Asset Address: `{asset_info['assetAddress']}`")
    st.write(f"aToken Address: `{asset_info['aTokenAddress']}`")
    st.write(f"Variable Debt Token Address: `{asset_info['variableDebtTokenAddress']}`")




# Create tabs for Collateral and Debt views
tab1, tab2 = st.tabs(["Collateral View", "Debt View"])

with tab1:
    st.header("Collateral Analysis")
    # Existing code for collateral analysis
    selected_asset = st.selectbox('Select a collateral asset:', list(collateral_data.keys()), key='collateral_select')
    
    if selected_asset in collateral_data:
        sorted_debt_symbols, sorted_values = collateral_data[selected_asset]
        fig, table_data = create_proportion_charts(selected_asset, sorted_values, sorted_debt_symbols, new_asset_data_df)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Debt Breakdown:")
        formatted_table_data = format_table_data(table_data)
        st.dataframe(formatted_table_data)

with tab2:
    st.header("Debt Analysis")
    # New code for debt analysis
    selected_debt = st.selectbox('Select a debt asset:', list(debt_data.keys()), key='debt_select')
    
    if selected_debt in debt_data:
        sorted_collateral_symbols, sorted_values = debt_data[selected_debt]
        fig, table_data = create_proportion_charts(selected_debt, sorted_values, sorted_collateral_symbols, new_asset_data_df)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Collateral Breakdown:")
        formatted_table_data = format_table_data(table_data)
        st.dataframe(formatted_table_data)

logger.info("Pie charts and tables have been created for both collateral and debt views")



# Create a mapping from new_asset_data_df
asset_mapping = new_asset_data_df.set_index('symbol')[['price', 'liquidationThreshold']].to_dict('index')

# Function to calculate total scaled collateral and total user debt
def calculate_user_metrics(row):
    total_scaled_collateral = 0
    total_actual_collateral = 0  # Initialize this variable
    total_user_debt = 0
    
    for symbol, data in asset_mapping.items():
        # Calculate scaled collateral
        collateral_col = f"a{symbol}"
        if collateral_col in row.index:
            total_scaled_collateral += row[collateral_col] * data['liquidationThreshold'] * data['price']
            total_actual_collateral += row[collateral_col] * data['price']
        
        # Calculate user debt
        debt_col = f"d{symbol}"
        if debt_col in row.index:
            total_user_debt += row[debt_col] * data['price']
    
    return pd.Series({
        'total_scaled_collateral': total_scaled_collateral,
        'total_actual_collateral': total_actual_collateral,
        'total_user_debt': total_user_debt
    })

# Apply the function to new_user_position_data
new_user_position_data[['total_scaled_collateral', 'total_actual_collateral', 'total_user_debt']] = new_user_position_data.apply(calculate_user_metrics, axis=1)

# Calculate health ratio
new_user_position_data['health_ratio'] = new_user_position_data['total_scaled_collateral'] / new_user_position_data['total_user_debt']

# Replace infinity values with a large number (for cases where total_user_debt is 0)
new_user_position_data['health_ratio'] = new_user_position_data['health_ratio'].replace([np.inf, -np.inf], 1e6)

# Handle NaN values (for cases where both total_scaled_collateral and total_user_debt are 0)
new_user_position_data['health_ratio'] = new_user_position_data['health_ratio'].fillna(0)

# Filter the dataframe for total_user_debt > 100
filtered_data = new_user_position_data[new_user_position_data['total_user_debt'] > 100]
# Filter for emode 0
filtered_data = filtered_data[filtered_data['emode'] == 0]

# Create the scatter plot: total_actual_collateral vs health_ratio
fig2 = px.scatter(filtered_data, 
                 x='health_ratio', 
                 y='total_actual_collateral', 
                 size='total_user_debt',
                 hover_data=['total_user_debt', 'health_ratio'],
                 labels={
                     'health_ratio': 'Health Ratio',
                     'total_actual_collateral': 'Total Actual Collateral',
                     'total_user_debt': 'Total User Debt'
                 },
                 title='User Positions: Actual Collateral vs Health Ratio (Debt > 100)')

# Update layout to set axis ranges and increase size
fig2.update_layout(
    xaxis_range=[0, 5],
    yaxis_range=[2, 8.5],
    height=1000,  # Increase height
    width=1000,  # Increase width
)

# Update y-axis to logarithmic scale
fig2.update_yaxes(type='log')

# Update marker properties
fig2.update_traces(
    marker=dict(
        color='blue',  # Set a solid color (you can change 'blue' to any color you prefer)
        line=dict(color='black', width=0),  # This sets the border
        opacity=0.5,   # Add some transparency
        sizemode='area',  # Scale size by area instead of diameter
        sizeref=2.*max(filtered_data['total_user_debt'])/(70**2),  # Scale factor for marker size
    )
)

# Display the plot
st.plotly_chart(fig2, use_container_width=True)

# new_user_position_data.to_csv("./data/user_position_data_with_metrics.csv", index=False)

# Add download buttons for CSV files
st.header("Download Data")

# Function to convert dataframe to CSV
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

# Download button for new_user_position_data
csv_user_position = convert_df(new_user_position_data)
st.download_button(
    label="Download User Position Data",
    data=csv_user_position,
    file_name="user_position_data.csv",
    mime="text/csv",
)

# Download button for new_asset_data_df
csv_asset_data = convert_df(new_asset_data_df)
st.download_button(
    label="Download Asset Data",
    data=csv_asset_data,
    file_name="asset_data.csv",
    mime="text/csv",
)

logger.info("Download buttons for CSV files have been added")