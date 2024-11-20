from bson.objectid import ObjectId
from datetime import datetime, timezone


class UserService:
    def __init__(self, db):
        self.collection = db.get_collection("users")

    def create(self, data):
        user = {
            "name": data["name"],
            "phone_number": data["phone_number"],
            "department": data["department"],
            "position": data["position"],
            "created_at": timezone.utcnow(),
            "updated_at": timezone.utcnow(),
        }
        result = self.collection.insert_one(user)
        return str(result.inserted_id)

    def get_by_id(self, user_id):
        user = self.collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        user["_id"] = str(user["_id"])
        return user

    def get_all(self):
        users = self.collection.find()
        return [self._format(user) for user in users]

    def update(self, user_id, data):
        data["updated_at"] = datetime.utcnow()
        result = self.collection.update_one({"_id": ObjectId(user_id)}, {"$set": data})
        if result.matched_count == 0:
            raise ValueError(f"User with ID {user_id} not found")
        return {"message": "User updated successfully"}

    def delete(self, user_id):
        result = self.collection.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            raise ValueError(f"User with ID {user_id} not found")
        return {"message": "User deleted successfully"}

    @staticmethod
    def _format(user):
        user["_id"] = str(user["_id"])
        return user
