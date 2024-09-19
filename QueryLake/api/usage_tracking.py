from ..database.sql_db_tables import UsageTally
from sqlmodel import Session, select, and_
from ..typing.config import AuthType
from .single_user_auth import get_user
from typing import Dict

from time import time
from datetime import datetime, timezone
import json
from copy import deepcopy


def increment_merge_dictionaries(base_dict : dict, increment_dict : dict):
    # print("Merging")
    # print(json.dumps(base_dict, indent=4))
    # print(json.dumps(increment_dict, indent=4))
    
    for key, value in increment_dict.items():
        if key in base_dict:
            
            if isinstance(value, (int, float)) and isinstance(base_dict[key], (int, float)):
                base_dict[key] += value
            elif isinstance(value, dict) and isinstance(base_dict[key], dict):
                base_dict[key] = increment_merge_dictionaries(base_dict[key], value)
            
        elif key not in base_dict:
            
            base_dict[key] = value

    # print("MERGED DICT")
    # print(json.dumps(base_dict, indent=4))
    return base_dict


def increment_usage_tally(
        database : Session,
        auth : AuthType,
        usage_increment : dict,
        api_key_id : str = None
    ) -> None:
    
    user_db_entry, user_auth = get_user(database, auth)
    
    current_time = datetime.now(timezone.utc)
    
    
    
    current_time = current_time.replace(second=0, microsecond=0, minute=0)
    current_time_unix_hour = int(current_time.timestamp())
    current_time = current_time.replace(hour=0)
    current_time_unix_day = int(current_time.timestamp())
    current_time = current_time.replace(day=1)
    current_time_unix_month = int(current_time.timestamp())
    
    
    current_timestamps = {
        "hour": current_time_unix_hour, 
        "day": current_time_unix_day, 
        "month": current_time_unix_month
    }
    # current_db_entries : Dict[str, UsageTally] = {}
    
    for time_key, time_value in current_timestamps.items():
        statement = database.exec(
            select(UsageTally).where(
                and_(
                    UsageTally.user_id == user_db_entry.id,
                    UsageTally.window == time_key,
                    UsageTally.start_timestamp == time_value,
                    *([UsageTally.api_key_id == api_key_id] if not api_key_id is None else [])
                )
            )
        )
        retrieved = statement.all()
        
        if len(retrieved) > 0:
            current_entry = retrieved[0]
        else:
            current_entry = UsageTally(
                user_id=user_db_entry.id,
                window=time_key,
                start_timestamp=time_value,
                value={},
                **({"api_key_id": api_key_id} if not api_key_id is None else {})
            )
            database.add(current_entry)
        
        # print("CURRENT ENTRY", json.dumps(current_entry.value, indent=4))
        
        # print("ADDING INCREMENT", json.dumps(usage_increment, indent=4))
        
        base_input = deepcopy(current_entry.value)
        new_value = increment_merge_dictionaries(base_input, usage_increment)
        current_entry.value = new_value
        
        # print("UPDATED CURRENT ENTRY", json.dumps(current_entry.value, indent=4))
        

    database.commit()
    
    
def get_usage_tally(
        database : Session,
        auth : AuthType,
        window : str,
        start_timestamp : int,
        end_timestamp : int = None
    ) -> dict:
    
    user_db_entry, user_auth = get_user(database, auth)
    
    assert window in ["hour", "day", "month"], "Invalid window; must be one of 'hour', 'day', 'month'"
    assert isinstance(start_timestamp, (int, float)), "start_timestamp must be an integer or float"
    assert isinstance(end_timestamp, (int, float)), "end_timestamp must be an integer or float"
    start_timestamp = int(start_timestamp)
    end_timestamp = int(end_timestamp)
    
    
    statement = database.exec(
        select(UsageTally).where(
            and_(
                UsageTally.user_id == user_db_entry.id,
                UsageTally.window == window,
                UsageTally.start_timestamp >= start_timestamp,
                UsageTally.start_timestamp <= end_timestamp
            )
            # UsageTally.user_id == user_db_entry.id
        )
    )
    retrieved = list(statement.all())
    
    if len(retrieved) > 0:
        return retrieved
    else:
        return []