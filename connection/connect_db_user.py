from databases import Database
import platform

def convert_to_linux_path(windows_path):
    # Replace backslashes with forward slashes
    linux_path = windows_path.replace("\\", "/")
    return linux_path

# paths
windows_path = "sqlite:///db_user\\users.db"
linux_path = convert_to_linux_path(windows_path)

# connect to the database
if platform.system() == "Windows":
    database_user = Database(windows_path)
else:
    database_user = Database(linux_path)


# function to query the chathistory table to get historical chat data
async def get_historical_chat_data(uid, session_id):
    query = "SELECT * FROM chathistory WHERE uid = :uid AND session_id = :session_id"
    values = {"uid": uid, "session_id": session_id}
    data =  await database_user.fetch_all(query=query, values=values)
    return data

# async def append_history_chat_data(uid,session_id):
#     query = "SELECT * FROM chathistory WHERE uid = :uid AND session_id = :session_id"
#     values = {"uid": uid, "session_id": session_id}
#     data =  await database_user.fetch_all(query=query, values=values)
#     return data