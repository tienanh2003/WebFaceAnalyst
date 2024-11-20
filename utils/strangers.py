from bson.objectid import ObjectId
from datetime import datetime


class StrangerService:
    def __init__(self, db):
        self.collection = db.get_collection("strangers")

    def create(self, data):
        stranger = {
            "name": data.get("name", "NoName"),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        result = self.collection.insert_one(stranger)
        return str(result.inserted_id)

    def get_by_id(self, stranger_id):
        stranger = self.collection.find_one({"_id": ObjectId(stranger_id)})
        if not stranger:
            raise ValueError(f"Stranger with ID {stranger_id} not found")
        stranger["_id"] = str(stranger["_id"])
        return stranger

    def get_all(self):
        strangers = self.collection.find()
        return [self._format(stranger) for stranger in strangers]

    def delete(self, stranger_id):
        result = self.collection.delete_one({"_id": ObjectId(stranger_id)})
        if result.deleted_count == 0:
            raise ValueError(f"Stranger with ID {stranger_id} not found")
        return {"message": "Stranger deleted successfully"}

    @staticmethod
    def _format(stranger):
        stranger["_id"] = str(stranger["_id"])
        return stranger
