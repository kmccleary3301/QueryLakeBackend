# import os
# from ..database import sql_db_tables
# from sqlmodel import Session, select, and_, not_
# from sqlalchemy.sql.operators import is_
# from .user_auth import *
# from .hashing import random_hash
# import json


# Get rid of this entire file. It is not used in the current version of the codebase.


# server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
# upper_server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])+"/"
# user_db_path = server_dir+"/user_db/files/"
# with open(upper_server_dir+"config.json", 'r', encoding='utf-8') as f:
#     file_read = f.read()
#     f.close()
# GLOBAL_SETTINGS = json.loads(file_read)


# def fetch_chat_sessions(database : Session, 
#                         username : str, 
#                         password_prehash : str,
#                         cutoff_date: str = None):
#     """
#     Get previous chat sessions of user. Returned as a list of objects sorted by timestamp.
#     """

#     user = get_user(database, username, password_prehash)

#     user_sessions = database.exec(select(sql_db_tables.chat_session_new).where(and_(sql_db_tables.chat_session_new.author_user_name == username, 
#                                                                                     not_(is_(sql_db_tables.chat_session_new.title, None)),
#                                                                                     sql_db_tables.chat_session_new.hidden == False))).all()
    
#     # print("sessions:", user_sessions)
#     user_sessions = sorted(user_sessions, key=lambda x: x.creation_timestamp)

#     return_sessions = []
#     for session in user_sessions:
#         return_sessions.append({
#             "time": session.creation_timestamp,
#             "title": session.title,
#             "hash_id": session.hash_id
#         })
#     # return {"success": True, "result": return_sessions[::-1]}
#     return return_sessions[::-1]

# def fetch_session(database : Session, 
#                     username : str, 
#                     password_prehash : str,
#                     hash_id: str):
#     """
#     Get all interactions from chat session by id.
#     """

#     user = get_user(database, username, password_prehash)

#     session = database.exec(select(sql_db_tables.chat_session_new).where(and_(sql_db_tables.chat_session_new.hash_id == hash_id, sql_db_tables.chat_session_new.author_user_name == username))).first()

#     bot_responses_previous = database.exec(select(sql_db_tables.chat_entry_model_response).where(sql_db_tables.chat_entry_model_response.chat_session_id == session.id)).all()
#     # print("Bot responses found:", bot_responses_previous)
#     bot_responses_previous = sorted(bot_responses_previous, key=lambda x: x.timestamp)

#     return_segments = []
#     for bot_response in bot_responses_previous:
#         question_previous = database.exec(select(sql_db_tables.chat_entry_user_question).where(sql_db_tables.chat_entry_user_question.id == bot_response.chat_entry_response_to)).first()
#         return_segments.append({"content": question_previous.content.encode("utf-8").hex(), "type": "user"})
#         if not bot_response.sources is None:
#             sources = [{"metadata": value} for value in json.loads(bot_response.sources)]
#             return_segments.append({"content": bot_response.content.encode("utf-8").hex(), "type": "bot", "sources": sources})
#         else:
#             return_segments.append({"content": bot_response.content.encode("utf-8").hex(), "type": "bot"})

#     # print("Got return segments:", return_segments)
#     # return {"success": True, "result": return_segments}
#     return return_segments

# def prune_empty_chat_sessions(database : Session):
#     """Get rid of empty chat sessions with no history."""

#     sessions = database.exec(select(sql_db_tables.chat_session_new).where(sql_db_tables.chat_session_new.title == None)).all()
#     for session in sessions:
#         database.delete(session)
#     database.commit()

# def create_chat_session(database : Session, 
#                         username : str, 
#                         password_prehash : str,
#                         access_token_hash_id : str = None,
#                         model_name : str = GLOBAL_SETTINGS["default_models"]["llm"]):
#     """Create a new chat session. Returns the hash_id of the created session."""


#     user = get_user(database, username, password_prehash)
#     if access_token_hash_id is None:
#         get_access_token = database.exec(select(sql_db_tables.access_token).where(sql_db_tables.access_token.author_user_name == username)).first()
#     else:
#         get_access_token = database.exec(select(sql_db_tables.access_token).where(sql_db_tables.access_token.hash_id == access_token_hash_id)).first()
    
#     session_hash = random_hash()
#     new_session = sql_db_tables.chat_session_new(
#         hash_id=session_hash,
#         model=model_name,
#         author_user_name=username,
#         creation_timestamp=time.time(),
#         access_token_id=get_access_token.id,
#     )

#     database.add(new_session)
#     database.commit()
#     # return {"success": True, "session_hash": session_hash}
#     return session_hash

# def hide_chat_session(database : Session, 
#                         username : str, 
#                         password_prehash : str,
#                         hash_id: str):
#     """
#     Permanently hide chat session so it does not show up in history.
#     """

#     user = get_user(database, username, password_prehash)

#     session = database.exec(select(sql_db_tables.chat_session_new).where(and_(sql_db_tables.chat_session_new.hash_id == hash_id, sql_db_tables.chat_session_new.author_user_name == username))).first()

#     session.hidden = True
#     database.commit()
#     # print("Got return segments:", return_segments)
#     # return {"success": True}
#     return True



