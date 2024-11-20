from bson.objectid import ObjectId
from datetime import datetime, timezone


class PersonAppearanceService:
    def __init__(self, db):
        self.collection = db.get_collection("person_appearances")

    def create(self, data):
        appearance = {
            "video_id": data["video_id"],
            "user_id": data.get("user_id"),
            "start_time": data["start_time"],
            "end_time": data["end_time"],
            "created_at": timezone.utcnow(),
        }
        result = self.collection.insert_one(appearance)
        return str(result.inserted_id)

    def get_by_id(self, appearance_id):
        appearance = self.collection.find_one({"_id": ObjectId(appearance_id)})
        if not appearance:
            raise ValueError(f"Appearance with ID {appearance_id} not found")
        appearance["_id"] = str(appearance["_id"])
        return appearance

    def get_all(self):
        appearances = self.collection.find()
        return [self._format(appearance) for appearance in appearances]

    def delete(self, appearance_id):
        result = self.collection.delete_one({"_id": ObjectId(appearance_id)})
        if result.deleted_count == 0:
            raise ValueError(f"Appearance with ID {appearance_id} not found")
        return {"message": "Appearance deleted successfully"}

    @staticmethod
    def _format(appearance):
        appearance["_id"] = str(appearance["_id"])
        return appearance
