from pymongo import MongoClient

# mongodb+srv://tienanh:tienanh@faceanalyst.xl7yw.mongodb.net/
url = "mongodb+srv://tienanh:tienanh@faceanalyst.xl7yw.mongodb.net/"

class Database:
    def __init__(self, uri=url, db_name="testdb"):

        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def get_collection(self, collection_name):
        return self.db[collection_name]

    def close_connection(self):
        self.client.close()
