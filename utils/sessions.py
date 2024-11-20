from bson.objectid import ObjectId
from datetime import datetime, timezone


class SessionService:
    def __init__(self, db):
        self.collection = db.get_collection("sessions")

    def create(self, data):
        session = {
            "user_id": data.get("user_id"),
            "start_time": data["start_time"],
            "end_time": data["end_time"],
            "location": data["location"],
            "created_at": timezone.utcnow(),
        }
        result = self.collection.insert_one(session)
        return str(result.inserted_id)

    def get_by_id(self, session_id):
        session = self.collection.find_one({"_id": ObjectId(session_id)})
        if not session:
            raise ValueError(f"Session with ID {session_id} not found")
        session["_id"] = str(session["_id"])
        return session

    def get_all(self):
        sessions = self.collection.find()
        return [self._format(session) for session in sessions]

    def delete(self, session_id):
        result = self.collection.delete_one({"_id": ObjectId(session_id)})
        if result.deleted_count == 0:
            raise ValueError(f"Session with ID {session_id} not found")
        return {"message": "Session deleted successfully"}

    @staticmethod
    def _format(session):
        session["_id"] = str(session["_id"])
        return session
