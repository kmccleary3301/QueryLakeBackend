from ..database.sql_db_tables import UsageTally
from sqlmodel import Session, select, and_
from ..typing.config import AuthType
from .single_user_auth import get_user
from typing import Dict

from time import time
from datetime import datetime, timezone


def increment_merge_dictionaries(base_dict, increment_dict):
    for key, value in increment_dict.items():
        if key in base_dict:
            
            if isinstance(value, (int, float)) and isinstance(base_dict[key], (int, float)):
                base_dict[key] += value
            elif isinstance(value, dict) and isinstance(base_dict[key], dict):
                base_dict[key] = increment_merge_dictionaries(base_dict[key], value)
            
        elif key not in base_dict:
            
            base_dict[key] = value

    return base_dict


def increment_usage_tally(
        database : Session,
        auth : AuthType,
        usage_increment : dict,
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
    current_db_entries : Dict[str, UsageTally] = {}
    
    for time_key, time_value in current_timestamps.items():
        statement = database.exec(
            select(UsageTally).where(
                and_(
                    UsageTally.user_id == user_db_entry.id,
                    UsageTally.window == time_key,
                    UsageTally.start_timestamp == time_value
                )
            )
        )
        retrieved = statement.all()
        
        if len(retrieved) > 0:
            current_db_entries[time_key] = retrieved[0]
        else:
            current_db_entries[time_key] = UsageTally(
                user_id=user_db_entry.id,
                window=time_key,
                start_timestamp=time_value,
            )
            database.add(current_db_entries[time_key])
            
    for window, window_entry in current_db_entries.items():
        window_entry.value = increment_merge_dictionaries(window_entry.value, usage_increment)

    database.commit()