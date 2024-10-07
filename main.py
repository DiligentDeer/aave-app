import pandas as pd
from web3 import Web3
from utils import get_new_asset_data, save_new_data, merge_and_save, get_current_unix_timestamp, get_user_data, get_user_position_data
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import locale
from plotly.subplots import make_subplots
import numpy as np
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the locale for number formatting
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

@st.cache_data
def load_initial_data():
    asset_data = pd.read_csv("./data/asset_data.csv")
    user_data = pd.read_csv("./data/user_data.csv")
    user_position_data = pd.read_csv("./data/user_position_data.csv")
    return asset_data, user_data, user_position_data

@st.cache_data
def process_data(asset_data, user_data, user_position_data):
    current_unix_timestamp = get_current_unix_timestamp()

    # Process asset data
    highest_timestamp_asset_data = asset_data["timestamp"].max()
    if highest_timestamp_asset_data + (86400/12) < current_unix_timestamp:
        logger.info("Fetching new asset data")
        new_asset_data = get_new_asset_data()
        new_asset_data_df = pd.DataFrame(new_asset_data)
        save_new_data(new_asset_data_df, "./data/asset_data.csv")
    else:
        logger.info("Using existing asset data")
        highest_timestamp_asset_data = asset_data["timestamp"].max()
        new_asset_data = asset_data[asset_data['timestamp'] == highest_timestamp_asset_data].to_dict(orient='records')
        new_asset_data_df = asset_data[asset_data['timestamp'] == highest_timestamp_asset_data]

    # print("Type of new_asset_data:", type(new_asset_data))
    # print("Number of items in new_asset_data:", len(new_asset_data))
    # print("First item in new_asset_data:", new_asset_data[0] if new_asset_data else "No data")
    
    # Process user data
    highest_timestamp_user_data = user_data["timestamp"].max()
    if highest_timestamp_user_data + 86400 < current_unix_timestamp:
        logger.info("Fetching new user data")
        new_user_data = get_user_data()
        save_new_data(new_user_data, "./data/user_data.csv")
    else:
        logger.info("Using existing user data")
        new_user_data = user_data[user_data['timestamp'] == highest_timestamp_user_data]

    # Process user position data
    user_addresses = new_user_data['user'].tolist()
    users_checksum = [Web3.to_checksum_address(user) for user in user_addresses]
    
    highest_timestamp_user_position_data = user_position_data["timestamp"].max()
    if highest_timestamp_user_position_data + 86400 < current_unix_timestamp:
        logger.info("Fetching new user position data")
        new_user_position_data = get_user_position_data(users_checksum, new_asset_data)
        save_new_data(new_user_position_data, "./data/user_position_data.csv")
    else:
        logger.info("Using existing user position data")
        new_user_position_data = user_position_data[user_position_data['timestamp'] == highest_timestamp_user_position_data]

    return new_asset_data_df, new_user_position_data


# @st.cache_data
# def prepare_data_for_prop(new_asset_data_df, new_user_position_data):
#     extracted_asset_list = new_asset_data_df['symbol'].tolist()
#     price_dict = dict(zip(new_asset_data_df['symbol'], new_asset_data_df['price']))

#     def get_price(column_name):
#         if column_name in ['user', 'timestamp']:
#             return 1
#         symbol = column_name[1:]
#         return price_dict.get(symbol, 1)

#     new_df = new_user_position_data.copy()
#     for column in new_user_position_data.columns:
#         if column not in ['user', 'timestamp']:
#             price = get_price(column)
#             new_df[f'{column}_value'] = new_user_position_data[column] * price
            
#     # Calculate sum of all 'a{symbol}_value' for each row
#     new_df['total_a_value'] = new_df[[f'a{symbol}_value' for symbol in extracted_asset_list if f'a{symbol}_value' in new_df.columns]].sum(axis=1)
    
#     # Calculate sum of all 'd{symbol}_value' for each row
#     new_df['total_d_value'] = new_df[[f'd{symbol}_value' for symbol in extracted_asset_list if f'd{symbol}_value' in new_df.columns]].sum(axis=1)
    
#     # Calculate proportions for each asset
#     for symbol in extracted_asset_list:
#         if f'a{symbol}_value' in new_df.columns:
#             new_df[f'a{symbol}_prop'] = new_df[f'a{symbol}_value'] / new_df['total_a_value']
#         if f'd{symbol}_value' in new_df.columns:
#             new_df[f'd{symbol}_prop'] = new_df[f'd{symbol}_value'] / new_df['total_d_value']
    
#     new_df_prop = new_df

#     return new_df_prop, extracted_asset_list

@st.cache_data
def prepare_data_for_visualization(new_asset_data_df, new_user_position_data):
    extracted_asset_list = new_asset_data_df['symbol'].tolist()
    price_dict = dict(zip(new_asset_data_df['symbol'], new_asset_data_df['price']))

    def get_price(column_name):
        if column_name in ['user', 'timestamp']:
            return 1
        symbol = column_name[1:]
        return price_dict.get(symbol, 1)

    new_df = new_user_position_data.copy()
    for column in new_user_position_data.columns:
        if column not in ['user', 'timestamp']:
            price = get_price(column)
            new_df[f'{column}_value'] = new_user_position_data[column] * price

    return new_df, extracted_asset_list

@st.cache_data
def prepare_collateral_debt_data(new_df, collateral_symbols, debt_symbols):
    collateral_data = {}
    for asset in collateral_symbols:
        heatmap_data = []
        mask = new_df[f'a{asset}_value'] > 10
        for debt in debt_symbols:
            if f'd{debt}' in new_df.columns:
                value = new_df.loc[mask, f'd{debt}_value'].sum()
            else:
                value = 0
            if value >= 100:
                heatmap_data.append((debt, value))
        if heatmap_data:
            heatmap_data.sort(key=lambda x: x[1], reverse=True)
            sorted_debt_symbols, sorted_values = zip(*heatmap_data)
            collateral_data[asset] = (sorted_debt_symbols, sorted_values)

    debt_data = {}
    for debt in debt_symbols:
        heatmap_data = []
        mask = new_df[f'd{debt}_value'] > 10
        for asset in collateral_symbols:
            if f'a{asset}' in new_df.columns:
                value = new_df.loc[mask, f'a{asset}_value'].sum()
            else:
                value = 0
            if value >= 100:
                heatmap_data.append((asset, value))
        if heatmap_data:
            heatmap_data.sort(key=lambda x: x[1], reverse=True)
            sorted_collateral_symbols, sorted_values = zip(*heatmap_data)
            debt_data[debt] = (sorted_collateral_symbols, sorted_values)

    return collateral_data, debt_data



# @st.cache_data
# def prepare_collateral_debt_data_prop(new_df, collateral_symbols, debt_symbols):
#     collateral_data = {}
#     for asset in collateral_symbols:
#         heatmap_data = []
#         mask = new_df[f'a{asset}_value'] > 10
#         for debt in debt_symbols:
#             if f'd{debt}' in new_df.columns and f'a{asset}_prop' in new_df.columns and f'd{debt}_prop' in new_df.columns:
#                 value = (new_df.loc[mask, f'a{debt}_prop'] * 
#                          new_df.loc[mask, f'd{debt}_prop'] * 
#                          new_df.loc[mask, f'd{debt}_value']).sum()
#             else:
#                 value = 0
#             if value >= 100:
#                 heatmap_data.append((debt, value))
#         if heatmap_data:
#             heatmap_data.sort(key=lambda x: x[1], reverse=True)
#             sorted_debt_symbols, sorted_values = zip(*heatmap_data)
#             collateral_data[asset] = (sorted_debt_symbols, sorted_values)

#     debt_data = {}
#     for debt in debt_symbols:
#         heatmap_data = []
#         mask = new_df[f'd{debt}_value'] > 10
#         for asset in collateral_symbols:
#             if f'a{asset}' in new_df.columns and f'a{asset}_prop' in new_df.columns and f'd{debt}_prop' in new_df.columns:
#                 value = (new_df.loc[mask, f'a{asset}_prop'] * 
#                          new_df.loc[mask, f'd{asset}_prop'] * 
#                          new_df.loc[mask, f'a{asset}_value']).sum()
#             else:
#                 value = 0
#             if value >= 100:
#                 heatmap_data.append((asset, value))
#         if heatmap_data:
#             heatmap_data.sort(key=lambda x: x[1], reverse=True)
#             sorted_collateral_symbols, sorted_values = zip(*heatmap_data)
#             debt_data[debt] = (sorted_collateral_symbols, sorted_values)

#     return collateral_data, debt_data



@st.cache_data
def create_proportion_charts(asset, data, sorted_debt_symbols):
    total = sum(data)
    threshold = 0.01
    debt_data = list(zip(sorted_debt_symbols, data, [val / total for val in data]))
    main_categories = [item for item in debt_data if item[2] >= threshold]
    others = [item for item in debt_data if item[2] < threshold]
    if others:
        others_value = sum(item[1] for item in others)
        others_proportion = sum(item[2] for item in others)
        main_categories_with_others = main_categories + [("Others", others_value, others_proportion)]
    else:
        main_categories_with_others = main_categories
    main_categories_with_others.sort(key=lambda x: x[1], reverse=True)
    labels_pie, values_pie, proportions_pie = zip(*main_categories_with_others)
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'xy'}]])
    fig.add_trace(go.Pie(
        labels=labels_pie,
        values=values_pie,
        textinfo='percent',
        hoverinfo='label+value+percent',
        marker=dict(colors=px.colors.qualitative.Pastel)
    ), 1, 1)
    fig.add_trace(go.Bar(
        x=[label for label in labels_pie if label != "Others"],
        y=[prop * 100 for label, prop in zip(labels_pie, proportions_pie) if label != "Others"],
        text=[f'{prop:.1%}' for label, prop in zip(labels_pie, proportions_pie) if label != "Others"],
        textposition='auto',
        marker_color=px.colors.qualitative.Pastel[:len(labels_pie)-1]
    ), 1, 2)
    fig.update_layout(
        title=f"Debt Proportion for {asset}",
        height=500,
        width=1200,
    )
    fig.update_yaxes(title_text='Proportion (%)', row=1, col=2)
    fig.update_xaxes(title_text='Debt Assets', row=1, col=2)
    return fig

# @st.cache_data
# def create_proportion_table(data, sorted_debt_symbols, new_asset_data):
#     total = sum(data)
#     threshold = 0.005
#     if isinstance(new_asset_data, list):
#         new_asset_data = pd.DataFrame(new_asset_data)
#     debt_data = list(zip(sorted_debt_symbols, data, [val / total for val in data]))
#     main_categories = [item for item in debt_data if item[2] >= threshold]
#     table_data = pd.DataFrame({
#         'Debt': [item[0] for item in main_categories],
#         'Value': [item[1] for item in main_categories],
#         'Proportion': [f'{item[2]:.2%}' for item in main_categories]
#     })
#     for label in table_data['Debt']:
#         asset_data = new_asset_data[new_asset_data['symbol'] == label]
#         if not asset_data.empty:
#             asset_data = asset_data.iloc[0]
#             table_data.loc[table_data['Debt'] == label, 'Borrow Cap'] = asset_data['borrowCap']
#             table_data.loc[table_data['Debt'] == label, '% of Borrow Cap'] = 100 * asset_data['debtSupply'] / asset_data['borrowCap'] if asset_data['borrowCap'] != 0 else 0
#             table_data.loc[table_data['Debt'] == label, 'Current Borrow'] = asset_data['debtSupply']
#             table_data.loc[table_data['Debt'] == label, 'Current Borrow $'] = asset_data['debtSupply'] * asset_data['price']
#     table_data['% of Current Borrow'] = table_data['Value'] / table_data['Current Borrow $'] * 100
#     return table_data

@st.cache_data
def create_proportion_table(data, sorted_symbols, new_asset_data, view_type):
    total = sum(data)
    threshold = 0.005
    if isinstance(new_asset_data, list):
        new_asset_data = pd.DataFrame(new_asset_data)
    
    asset_data = list(zip(sorted_symbols, data, [val / total for val in data]))
    main_categories = [item for item in asset_data if item[2] >= threshold]
    
    if view_type == 'collateral':
        table_data = pd.DataFrame({
            'Debt': [item[0] for item in main_categories],
            'Value': [item[1] for item in main_categories],
            'Proportion': [f'{item[2]:.2%}' for item in main_categories]
        })
    else:  # debt view
        table_data = pd.DataFrame({
            'Collateral': [item[0] for item in main_categories],
            'Value': [item[1] for item in main_categories],
            'Proportion': [f'{item[2]:.2%}' for item in main_categories]
        })

    for label in table_data.iloc[:, 0]:  # first column, either 'Debt' or 'Collateral'
        asset_data = new_asset_data[new_asset_data['symbol'] == label]
        if not asset_data.empty:
            asset_data = asset_data.iloc[0]
            if view_type == 'collateral':
                table_data.loc[table_data['Debt'] == label, 'Borrow Cap'] = asset_data['borrowCap']
                table_data.loc[table_data['Debt'] == label, '% of Borrow Cap'] = 100 * asset_data['debtSupply'] / asset_data['borrowCap'] if asset_data['borrowCap'] != 0 else 0
                table_data.loc[table_data['Debt'] == label, 'Current Borrow'] = asset_data['debtSupply']
                table_data.loc[table_data['Debt'] == label, 'Current Borrow $'] = asset_data['debtSupply'] * asset_data['price']
            else:  # debt view
                table_data.loc[table_data['Collateral'] == label, 'Supply Cap'] = asset_data['supplyCap']
                table_data.loc[table_data['Collateral'] == label, '% of Supply Cap'] = 100 * asset_data['collateralSupply'] / asset_data['supplyCap'] if asset_data['supplyCap'] != 0 else 0
                table_data.loc[table_data['Collateral'] == label, 'Current Supply'] = asset_data['collateralSupply']
                table_data.loc[table_data['Collateral'] == label, 'Current Supply $'] = asset_data['collateralSupply'] * asset_data['price']

    if view_type == 'collateral':
        table_data['% of Current Borrow'] = table_data['Value'] / table_data['Current Borrow $'] * 100
    else:  # debt view
        table_data['% of Current Supply'] = table_data['Value'] / table_data['Current Supply $'] * 100

    return table_data

def format_table_data(table_data):
    def format_value(val, column):
        if pd.isna(val) or val == '':
            return 'N/A'
        try:
            float_val = float(val)
            if column in ['Value', 'Current Borrow $']:
                return f'${float_val:,.0f}'
            elif column in ['% of Borrow Cap', '% of Current Borrow']:
                return f'{float_val:.2f}%'
            else:
                return f'{float_val:,.0f}'
        except (ValueError, TypeError):
            return str(val)
    
    formatted_table_data = table_data.copy()
    for col in formatted_table_data.columns:
        if col not in ['Debt', 'Proportion']:
            formatted_table_data[col] = formatted_table_data[col].apply(lambda x: format_value(x, col))
    return formatted_table_data


@st.cache_data
def create_modified_asset_mapping(new_asset_data_df, selected_assets, modification_type, modification_values):
    modified_asset_data = new_asset_data_df.copy()
    for asset, value in zip(selected_assets, modification_values):
        if modification_type == 'liquidationThreshold':
            modified_asset_data.loc[modified_asset_data['symbol'] == asset, 'liquidationThreshold'] = value
        elif modification_type == 'price':
            modified_asset_data.loc[modified_asset_data['symbol'] == asset, 'price'] = value
    return modified_asset_data.set_index('symbol')[['price', 'liquidationThreshold']].to_dict('index')



@st.cache_data
def prepare_health_ratio_data(new_user_position_data, asset_mapping):
    def calculate_user_metrics(row):
        total_scaled_collateral = 0
        total_actual_collateral = 0
        total_user_debt = 0
        for symbol, data in asset_mapping.items():
            collateral_col = f"a{symbol}"
            if collateral_col in row.index:
                total_scaled_collateral += row[collateral_col] * data['liquidationThreshold'] * data['price']
                total_actual_collateral += row[collateral_col] * data['price']
            debt_col = f"d{symbol}"
            if debt_col in row.index:
                total_user_debt += row[debt_col] * data['price']
        return pd.Series({
            'total_scaled_collateral': total_scaled_collateral,
            'total_actual_collateral': total_actual_collateral,
            'total_user_debt': total_user_debt
        })

    new_user_position_data[['total_scaled_collateral', 'total_actual_collateral', 'total_user_debt']] = new_user_position_data.apply(calculate_user_metrics, axis=1)
    new_user_position_data['health_ratio'] = new_user_position_data['total_scaled_collateral'] / new_user_position_data['total_user_debt']
    new_user_position_data['health_ratio'] = new_user_position_data['health_ratio'].replace([np.inf, -np.inf], 1e6)
    new_user_position_data['health_ratio'] = new_user_position_data['health_ratio'].fillna(0)
    filtered_data = new_user_position_data[(new_user_position_data['total_user_debt'] > 100) & (new_user_position_data['emode'] == 0)]
    sorted_data = filtered_data.sort_values('health_ratio')
    sorted_data['cumulative_collateral'] = sorted_data['total_actual_collateral'].cumsum()
    return sorted_data

@st.cache_data
def create_health_ratio_chart(sorted_data):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sorted_data['health_ratio'],
        y=sorted_data['cumulative_collateral'],
        fill='tozeroy',
        fillcolor='rgba(0, 100, 80, 0.2)',
        line=dict(color='rgb(0, 100, 80)', width=2),
        name='Cumulative Collateral',
        hovertemplate='<b>Health Ratio</b>: %{x:.2f}' +
                      '<br><b>Cumulative Collateral</b>: $%{y:,.2f}' +
                      '<br><b>Collateral Added</b>: $%{customdata:,.2f}<extra></extra>',
        customdata=sorted_data['total_actual_collateral']
    ))
    
    for ratio in [1, 1.5]:
        fig2.add_vline(x=ratio, line_dash="dash", line_color="red", opacity=0.5)
        
        # Find the y-position for the annotation
        mask = sorted_data['health_ratio'] >= ratio
        if mask.any():
            y_position = sorted_data.loc[mask, 'cumulative_collateral'].iloc[0]
        else:
            y_position = sorted_data['cumulative_collateral'].max()
        
        fig2.add_annotation(x=ratio, y=y_position, text=f"HR = {ratio}", showarrow=True, arrowhead=2, arrowcolor="black")
    
    fig2.update_layout(
        title='User Positions: Cumulative Collateral vs Health Ratio',
        xaxis_title='Health Ratio',
        yaxis_title='Cumulative Collateral ($)',
        xaxis_range=[0, 2.5],
        height=600,
        width=1000,
        hovermode='x unified'
    )
    fig2.update_yaxes(type='log')
    return fig2

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def create_health_ratio_breakdown(sorted_data):
    ranges = [0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, float('inf')]
    labels = ['< 1.0', '1.0 - 1.1', '1.1 - 1.2', '1.2 - 1.3', '1.3 - 1.4', '1.4 - 1.5', '≥ 1.5']
    
    breakdown = []
    total_collateral = sorted_data['total_actual_collateral'].sum()
    cumulative_collateral = 0
    cumulative_percentage = 0
    
    for i in range(len(ranges) - 1):
        mask = (sorted_data['health_ratio'] >= ranges[i]) & (sorted_data['health_ratio'] < ranges[i+1])
        collateral = sorted_data.loc[mask, 'total_actual_collateral'].sum()
        cumulative_collateral += collateral
        percentage = collateral / total_collateral
        cumulative_percentage += percentage
        
        breakdown.append({
            'HF_Range': labels[i],
            'Collateral': collateral,
            'Percentage': percentage,
            'Cumulative Collateral': cumulative_collateral,
            'Cumulative Percentage': cumulative_percentage
        })
    
    return pd.DataFrame(breakdown)

def main():
    st.set_page_config(layout="wide")
    st.title('AAVE V3 - Dashboard & Sandbox')
    st.write("""
    _This app provides a comprehensive overview of the Aave V3 protocol on Ethereum chain, including detailed analysis of collateral and debt positions, 
    customizable asset parameter modifications, and a user-friendly interface for downloading and interacting with data._
    """)
    st.write("_**NOTE**: The dashboard is limited to the most recent snapshot of the data available_")
    st.write("_**PS**: The dashboard can take upto 10-15 mins to fetch onchain data (~35 Assets * ~40k Users * 2 categories = ~2.8M data points) and make visualizations. I appreciate your patience :)_")

    asset_data, user_data, user_position_data = load_initial_data()
    new_asset_data_df, new_user_position_data = process_data(asset_data, user_data, user_position_data)
    new_df, extracted_asset_list = prepare_data_for_visualization(new_asset_data_df, new_user_position_data)
    
    # new_df.to_csv("new_df.csv")
    
    collateral_symbols = [col[1:] for col in new_df.columns if col.startswith('a') and not col.endswith('_value') and not col.endswith('_prop')]
    debt_symbols = [col[1:] for col in new_df.columns if col.startswith('d') and not col.endswith('_value') and not col.endswith('_prop')]

    collateral_data, debt_data = prepare_collateral_debt_data(new_df, collateral_symbols, debt_symbols)

    # print(json.dumps(collateral_data, indent=4))
    # print(type(collateral_data))
    

    # Detailed Information Section
    st.header("Detailed Asset Information")
    selected_asset_info = st.selectbox('Select an asset:', new_asset_data_df['symbol'].tolist(), key='asset_info_select')
    
    if selected_asset_info:
        asset_info = new_asset_data_df[new_asset_data_df['symbol'] == selected_asset_info].iloc[0]
        
        collateral_supply_value = asset_info['collateralSupply'] * asset_info['price']
        debt_supply_value = asset_info['debtSupply'] * asset_info['price']
        utilization_rate = debt_supply_value / collateral_supply_value if collateral_supply_value > 0 else 0
        borrow_cap_ratio = asset_info['debtSupply'] / asset_info['borrowCap']
        supply_cap_ratio = asset_info['collateralSupply'] / asset_info['supplyCap']
        
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
        
        st.subheader("Asset Addresses")
        st.write(f"Asset Address: `{asset_info['assetAddress']}`")
        st.write(f"aToken Address: `{asset_info['aTokenAddress']}`")
        st.write(f"Variable Debt Token Address: `{asset_info['variableDebtTokenAddress']}`")

    st.header("Borrow and Supply Breakdown")
    st.write("""
    _This section provides a detailed breakdown of the borrow and supply positions for the selected asset. 
    It shows the proportion of the total borrow and supply for each asset, allowing users to understand 
    the distribution of these positions across different assets._
    """)


    # Collateral and Debt Analysis Tabs
    tab1, tab2 = st.tabs(["Collateral View", "Debt View"])

    with tab1:
        st.write("### Collateral Asset Analysis")
        
        st.write("""_The section shows the proportion of assets borrow against the chosen collateral asset_""")
        
        selected_asset = st.selectbox('Select a collateral asset:', list(collateral_data.keys()), key='collateral_select')
        
        st.write("""_The pie chart shown here only considers proportions above 1% and groups the rest into "Others"._""")
        
        if selected_asset in collateral_data:
            sorted_debt_symbols, sorted_values = collateral_data[selected_asset]
            fig = create_proportion_charts(selected_asset, sorted_values, sorted_debt_symbols)
            # In the Collateral View tab:
            table_data = create_proportion_table(sorted_values, sorted_debt_symbols, new_asset_data_df, 'collateral')

            st.plotly_chart(fig, use_container_width=True)
            
            st.write("#### Debt Breakdown:")
            st.write("""
            _This table breaks down the debt composition for the selected asset. Key points:_

            - _Shows assets representing ≥0.5% of total composition_
            - _'Value' and 'Proportion' columns show distribution across assets_
            - _'Borrow Cap' and '% of Borrow Cap' indicate borrowing limits and utilization_
            - _'Current Borrow' columns show actual borrowing in units and USD_
            - _'% of Current Borrow' relates this asset's borrowing to overall borrowing_

            _Use this to understand asset distribution, identify high-utilization assets, and assess borrow cap proximity._
            """)
            formatted_table_data = format_table_data(table_data)
            st.dataframe(formatted_table_data)

    with tab2:
        st.write("### Borrow Asset Analysis")
        
        st.write("""_The section shows the proportion of assets used as collateral to back the chosen debt asset_""")
        
        selected_debt = st.selectbox('Select a debt asset:', list(debt_data.keys()), key='debt_select')
        
        st.write("""_The pie chart shown here only considers proportions above 1% and groups the rest into "Others"._""")
        
        if selected_debt in debt_data:
            sorted_collateral_symbols, sorted_values = debt_data[selected_debt]
            fig = create_proportion_charts(selected_debt, sorted_values, sorted_collateral_symbols)
            # In the Debt View tab:
            table_data = create_proportion_table(sorted_values, sorted_collateral_symbols, new_asset_data_df, 'debt')
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("#### Collateral Breakdown:")
            
            st.write("""
            _This table breaks down the collateral composition for the selected asset. Key points:_

            - _Shows assets representing ≥0.5% of total composition_
            - _'Value' and 'Proportion' columns show distribution across assets_
            - _'Borrow Cap' and '% of Borrow Cap' indicate borrowing limits and utilization_
            - _'Current Borrow' columns show actual borrowing in units and USD_
            - _'% of Current Borrow' relates this asset's borrowing to overall borrowing_

            _Use this to understand asset distribution, identify high-utilization assets, and assess borrow cap proximity._
            """)
            
            formatted_table_data = format_table_data(table_data)
            st.dataframe(formatted_table_data)

    st.header("Current User Position Status & Sandbox")

    st.write("""
    _This section allows you to simulate changes in asset parameters and visualize their impact on the protocol's health ratios. 
    Select assets, modify their liquidation threshold or price, and see how it affects the overall risk profile._
    """)

    selected_assets = st.multiselect("Select assets to modify:", new_asset_data_df['symbol'].tolist())
    modification_type = st.radio("Choose parameter to modify:", ['liquidationThreshold', 'price'])

    modification_values = []
    for asset in selected_assets:
        current_value = new_asset_data_df[new_asset_data_df['symbol'] == asset][modification_type].values[0]
        new_value = st.number_input(f"New {modification_type} for {asset}", value=float(current_value), format="%.4f")
        modification_values.append(new_value)

    # Display the default chart
    default_sorted_data = prepare_health_ratio_data(new_df, new_asset_data_df.set_index('symbol')[['price', 'liquidationThreshold']].to_dict('index'))
    default_fig = create_health_ratio_chart(default_sorted_data)
    st.plotly_chart(default_fig, use_container_width=True)

    # Create and display the default health ratio breakdown table
    default_breakdown_df = create_health_ratio_breakdown(default_sorted_data)
    st.write("### Current Health Ratio Analysis:")
    st.dataframe(default_breakdown_df.style.format({
        'Collateral': '${:,.2f}',
        'Percentage': '{:.2%}',
        'Cumulative Collateral': '${:,.2f}',
        'Cumulative Percentage': '{:.2%}'
    }))
    
    if st.button("Update Health Ratio Chart"):
        modified_asset_mapping = create_modified_asset_mapping(new_asset_data_df, selected_assets, modification_type, modification_values)
        modified_sorted_data = prepare_health_ratio_data(new_df, modified_asset_mapping)
        modified_fig = create_health_ratio_chart(modified_sorted_data)
        st.plotly_chart(modified_fig, use_container_width=True)

        # Create and display the health ratio breakdown table
        breakdown_df = create_health_ratio_breakdown(modified_sorted_data)
        
        st.write("### Health Ratio Analysis - After Modifications:")
        st.write("""
        _This table shows how the total collateral is distributed across different health ratio ranges. 
        A health ratio below 1.0 indicates a higher risk of liquidation. The 'Cumulative' columns show 
        the total collateral and percentage up to and including each range, helping to identify potential 
        risk thresholds in the modified scenario._
        _**NOTE**: The values deemed to be liquidated does not consider the CLOSE_FACTOR._
        """)
        st.dataframe(breakdown_df.style.format({
            'Collateral': '${:,.2f}',
            'Percentage': '{:.2%}',
            'Cumulative Collateral': '${:,.2f}',
            'Cumulative Percentage': '{:.2%}'
        }))

        # Calculate and display some key metrics
        total_collateral = breakdown_df['Collateral'].sum()
        collateral_below_1 = breakdown_df.loc[breakdown_df['HF_Range'] == '< 1.0', 'Collateral'].values[0]
        collateral_below_1_1 = breakdown_df.loc[breakdown_df['HF_Range'] != '≥ 1.1', 'Collateral'].sum()

        st.write(f"**Total Collateral**: ${total_collateral:,.2f}")
        st.write(f"**Collateral with Health Ratio < 1.0**: ${collateral_below_1:,.2f} ({collateral_below_1/total_collateral:.2%})")
        st.write(f"**Collateral with Health Ratio < 1.1**: ${collateral_below_1_1:,.2f} ({collateral_below_1_1/total_collateral:.2%})")


    # Download buttons
    st.write("### Download Data")
    csv_user_position = convert_df(new_df)
    st.download_button(
        label="Download User Position Data",
        data=csv_user_position,
        file_name="user_position_data.csv",
        mime="text/csv",
    )

    csv_asset_data = convert_df(new_asset_data_df)
    st.download_button(
        label="Download Asset Data",
        data=csv_asset_data,
        file_name="asset_data.csv",
        mime="text/csv",
    )

    logger.info("Streamlit app execution completed")

if __name__ == "__main__":
    main()
    