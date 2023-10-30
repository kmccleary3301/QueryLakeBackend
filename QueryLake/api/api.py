from .chat_sessions import *
from .collections import *
from .document import *
from .hashing import *
from .organizations import *
from .user_auth import *

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
print(server_dir)
if not os.path.exists(server_dir+"/user_db"):
    os.mkdir(server_dir+"/user_db")
if not os.path.exists(server_dir+"/user_db/files"):
    os.mkdir(server_dir+"/user_db/files")
user_db_path = server_dir+"/user_db/files/"

excluded_member_function_descriptions = ["and_", "is_", "not_", "select"]
        