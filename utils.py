##### Import libraries #####
from web3 import Web3, HTTPProvider
import pandas as pd
from requests import get, post
import time
import datetime
import os
import random
import ast
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union
from functools import lru_cache
import json

from dune_client.client import DuneClient
from dune_client.query import QueryBase

##### Environment variables #####
load_dotenv()

ALCHEMY_KEY = os.getenv("ALCHEMY_KEY_2")
INFURA_KEY = os.getenv("INFURA_KEY")

##### Declare constants #####
BAL_ADDRESS = "0x5c438e0e82607a3a07e6726b10e200739635895b"
ABI_BAL = [{"inputs":[{"internalType":"address","name":"proxyContractAddress","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[{"internalType":"address[]","name":"addresses","type":"address[]"}],"name":"batchUserEMode","outputs":[{"internalType":"uint256[]","name":"","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address[]","name":"addresses","type":"address[]"},{"internalType":"address","name":"tokenAddress","type":"address"}],"name":"checkBalances","outputs":[{"internalType":"uint256[]","name":"","type":"uint256[]"}],"stateMutability":"view","type":"function"}]

POOL_ADDRESS = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
ABI_POOL = [
    {
        "inputs": [
            {
                "internalType": "address",
                "name": "asset",
                "type": "address"
            }
        ],
        "name": "getReserveData",
        "outputs": [
            {
                "components": [
                    {
                        "components": [
                            {
                                "internalType": "uint256",
                                "name": "data",
                                "type": "uint256"
                            }
                        ],
                        "internalType": "struct DataTypes.ReserveConfigurationMap",
                        "name": "configuration",
                        "type": "tuple"
                    },
                    {
                        "internalType": "uint128",
                        "name": "liquidityIndex",
                        "type": "uint128"
                    },
                    {
                        "internalType": "uint128",
                        "name": "currentLiquidityRate",
                        "type": "uint128"
                    },
                    {
                        "internalType": "uint128",
                        "name": "variableBorrowIndex",
                        "type": "uint128"
                    },
                    {
                        "internalType": "uint128",
                        "name": "currentVariableBorrowRate",
                        "type": "uint128"
                    },
                    {
                        "internalType": "uint128",
                        "name": "currentStableBorrowRate",
                        "type": "uint128"
                    },
                    {
                        "internalType": "uint40",
                        "name": "lastUpdateTimestamp",
                        "type": "uint40"
                    },
                    {
                        "internalType": "uint16",
                        "name": "id",
                        "type": "uint16"
                    },
                    {
                        "internalType": "address",
                        "name": "aTokenAddress",
                        "type": "address"
                    },
                    {
                        "internalType": "address",
                        "name": "stableDebtTokenAddress",
                        "type": "address"
                    },
                    {
                        "internalType": "address",
                        "name": "variableDebtTokenAddress",
                        "type": "address"
                    },
                    {
                        "internalType": "address",
                        "name": "interestRateStrategyAddress",
                        "type": "address"
                    },
                    {
                        "internalType": "uint128",
                        "name": "accruedToTreasury",
                        "type": "uint128"
                    },
                    {
                        "internalType": "uint128",
                        "name": "unbacked",
                        "type": "uint128"
                    },
                    {
                        "internalType": "uint128",
                        "name": "isolationModeTotalDebt",
                        "type": "uint128"
                    }
                ],
                "internalType": "struct DataTypes.ReserveDataLegacy",
                "name": "arg_0",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },{"inputs":[{"internalType":"address","name":"user","type":"address"}],"name":"getUserEMode","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]

DATA_PROVIDER = "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3"
ABI_RESERVE = [
    {
        "inputs": [],
        "name": "getAllReservesTokens",
        "outputs": [
            {
                "components": [
                    {"internalType": "string", "name": "symbol", "type": "string"},
                    {"internalType": "address", "name": "tokenAddress", "type": "address"}
                ],
                "internalType": "struct IPoolDataProvider.TokenData[]",
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

##### Declare variables #####
ALCHEMY_URL = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"
INFURA_URL = f"https://mainnet.infura.io/v3/{INFURA_KEY}"

W3 = Web3(HTTPProvider(ALCHEMY_URL))
W33 = Web3(HTTPProvider(INFURA_URL))


##### Declare functions #####
@lru_cache(maxsize=None)
def get_current_unix_timestamp() -> int:
    return int(time.time())
TIME_STAMP = get_current_unix_timestamp()

@lru_cache(maxsize=None)
def get_reserve_list(block_number: Optional[int] = None) -> List[Dict[str, Union[str, int]]]:
    
    try:
        # Create contract instance
        contract = W3.eth.contract(address=Web3.to_checksum_address(DATA_PROVIDER), abi=ABI_RESERVE)
        
        # Get the contract function
        contract_function = getattr(contract.functions, "getAllReservesTokens")
        
        # Call the function with arguments and block identifier if provided
        if block_number is not None:
            data = contract_function().call(block_identifier=int(block_number))
        else:
            data = contract_function().call()
        
        return data
    
    except Exception as e:
        return f'Error querying smart contract: {e}'
    
@lru_cache(maxsize=None)
def get_asset_data(asset: str, block_number: Optional[int] = None) -> Dict[str, Union[str, int]]:
    
    try:
        # Create contract instance
        contract = W3.eth.contract(address=Web3.to_checksum_address(POOL_ADDRESS), abi=ABI_POOL)
        
        # Get the contract function
        contract_function = getattr(contract.functions, "getReserveData")
        
        # Call the function with arguments and block identifier if provided
        if block_number is not None:
            data = contract_function(asset).call(block_identifier=int(block_number))
        else:
            data = contract_function(asset).call()
        
        return data
    
    except Exception as e:
        return f'Error querying smart contract: {e}'
    
    
def decode_reserve_configuration(config_value: int) -> Dict[str, Union[float, bool]]:
    config = int(config_value)
    
    def get_bits(start: int, end: int) -> int:
        mask = (1 << (end - start + 1)) - 1
        return (config >> start) & mask
    
    return {
        "ltv": get_bits(0, 15)/10000,
        "liquidationThreshold": get_bits(16, 31)/10000,
        "liquidationBonus": get_bits(32, 47)/10000,
        "decimals": get_bits(48, 55),
        "reserveIsActive": bool(get_bits(56, 56)),
        "reserveIsFrozen": bool(get_bits(57, 57)),
        "borrowingEnabled": bool(get_bits(58, 58)),
        "stableRateBorrowingEnabled": bool(get_bits(59, 59)),
        "assetIsPaused": bool(get_bits(60, 60)),
        "borrowingInIsolationModeEnabled": bool(get_bits(61, 61)),
        "siloedBorrowingEnabled": bool(get_bits(62, 62)),
        "flashLoaningEnabled": bool(get_bits(63, 63)),
        "reserveFactor": get_bits(64, 79)/10000,
        "borrowCap": get_bits(80, 115),
        "supplyCap": get_bits(116, 151),
        "liquidationProtocolFee": get_bits(152, 167)/10000,
        "eModeCategoryId": get_bits(168, 175),
        "unbackedMintCap": get_bits(176, 211),
        "debtCeilingForIsolationMode": get_bits(212, 251),
        "virtualAccountingEnabled": bool(get_bits(252, 252))
    }
    
@lru_cache(maxsize=None)   
def get_current_price(token_address: str) -> Optional[float]:
    base_url = "https://coins.llama.fi/prices/current"
    url = f"{base_url}/ethereum:{token_address}"
    
    try:
        response = get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        price_info = data.get('coins', {}).get(f'ethereum:{token_address}', {})
        return price_info.get('price')
    except Exception as e:
        print(f"Error fetching price for {token_address}: {e}")
        return None
    




def merge_and_save(previous_data: pd.DataFrame, new_data: pd.DataFrame, path: str) -> None:
    # Concat data
    combined_data = pd.concat([previous_data, new_data], ignore_index=True)
    
    # Save data
    combined_data.to_csv(path, index=False)



def get_user_balance(user_addrress_list, asset, block_number=None):
    
    try:
        # Create contract instance
        contract = W33.eth.contract(address=Web3.to_checksum_address(BAL_ADDRESS), abi=ABI_BAL)
        
        # Get the contract function
        contract_function = getattr(contract.functions, "checkBalances")
        
        # Call the function with arguments and block identifier if provided
        if block_number is not None:
            data = contract_function(user_addrress_list,asset).call(block_identifier=int(block_number))
        else:
            data = contract_function(user_addrress_list,asset).call()
        
        return data
    
    except Exception as e:
        return f'Error querying smart contract: {e}'
    
@lru_cache(maxsize=None)
def get_supply(asset, block_number=None):
    
    SUPPLY_ABI = [{"inputs":[],"name":"totalSupply","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]
    try:
        # Create contract instance
        contract = W33.eth.contract(address=Web3.to_checksum_address(asset), abi=SUPPLY_ABI)
        
        # Get the contract function
        contract_function = getattr(contract.functions, "totalSupply")
        
        # Call the function with arguments and block identifier if provided
        if block_number is not None:
            data = contract_function().call(block_identifier=int(block_number))
        else:
            data = contract_function().call()
        
        return data
    
    except Exception as e:
        return f'Error querying smart contract: {e}'
    
def get_emode(user_list, block_number=None):
    
    try:
        # Create contract instance
        contract = W33.eth.contract(address=Web3.to_checksum_address(BAL_ADDRESS), abi=ABI_BAL)
        
        # Get the contract function
        contract_function = getattr(contract.functions, "batchUserEMode")
        
        # Call the function with arguments and block identifier if provided
        if block_number is not None:
            data = contract_function(user_list).call(block_identifier=int(block_number))
        else:
            data = contract_function(user_list).call()
        
        return data
    
    except Exception as e:
        return f'Error querying smart contract: {e}'
    
##### Function in Functions #####

@lru_cache(maxsize=None)
def get_new_asset_data() -> List[Dict[str, Union[int, str, float]]]:
    asset_list = get_reserve_list()
    
    data: List[Dict[str, Union[int, str, float]]] = []
    unix_timestamp: int = TIME_STAMP

    for asset in asset_list:
        data_dict: Dict[str, Union[int, str, float]] = {}
        
        reserve_data = get_asset_data(asset[1])
        data_dict["timestamp"] = unix_timestamp
        data_dict["symbol"] = asset[0]
        data_dict["price"] = get_current_price(asset[1])
        data_dict["assetAddress"] = asset[1]
        data_dict["configuration"] = reserve_data[0][0]

        data_dict["aTokenAddress"] = reserve_data[8]
        data_dict["stableDebtTokenAddress"] = reserve_data[9]
        data_dict["variableDebtTokenAddress"] = reserve_data[10]
        
        decoded_configuration = decode_reserve_configuration(reserve_data[0][0])
        
        data_dict["collateralSupply"] = get_supply(reserve_data[8]) / 10**decoded_configuration["decimals"]
        data_dict["debtSupply"] = get_supply(reserve_data[10]) / 10**decoded_configuration["decimals"]
        
        data_dict.update(decoded_configuration)
        data.append(data_dict)
        
    return data

@lru_cache(maxsize=None)
def get_user_data() -> pd.DataFrame:
    dune = DuneClient(
        api_key=os.getenv('DUNE_API_KEY2'),
        base_url="https://api.dune.com",
        request_timeout=5000 # request will time out after 300 seconds
    )

    query = QueryBase(
        query_id=4101003,
        params=[],
    )
    
    users = dune.run_query_dataframe(query)
    users["timestamp"] = TIME_STAMP

    return users

def get_user_position_data(users_checksum, asset_data) -> pd.DataFrame:
    
    if isinstance(asset_data, str):
        asset_data = json.loads(asset_data)
    elif not isinstance(asset_data, list):
        raise ValueError(f"Expected asset_data to be a list or JSON string, got {type(asset_data)}")

    user_position = []
    
    for user in users_checksum:
        user_position_dict = {"user": user}
        user_position_dict["timestamp"] = TIME_STAMP
        # user_position_dict["emode"] = get_emode(user)
        user_position.append(user_position_dict)

    emode = get_emode(users_checksum)
    for index, user_dict in enumerate(user_position):
        user_dict["emode"] = emode[index]     
        
    for asset in asset_data:
        decimals = asset["decimals"]
        
        atoken_symbol = f"a{asset['symbol']}"
        dtoken_symbol = f"d{asset['symbol']}"
        
        atoken_balance = get_user_balance(users_checksum, asset["aTokenAddress"])
        dtoken_balance = get_user_balance(users_checksum, asset["variableDebtTokenAddress"])
        
        for index, user_dict in enumerate(user_position):
            user_dict[atoken_symbol] = atoken_balance[index] / 10**decimals
            user_dict[dtoken_symbol] = dtoken_balance[index] / 10**decimals
            
    # Convert this to a dataframe
    df_user_position = pd.DataFrame(user_position)
    
    return df_user_position