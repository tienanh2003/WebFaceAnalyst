from bson.objectid import ObjectId
from datetime import datetime, timezone


class EmotionService:
    def __init__(self, db):
        self.collection = db.get_collection("emotions")

    def create(self, data):
        emotion = {
            "emotion_name": data["emotion_name"],
            "description": data["description"],
            "created_at": timezone.utcnow(),
        }
        result = self.collection.insert_one(emotion)
        return str(result.inserted_id)

    def get_by_id(self, emotion_id):
        emotion = self.collection.find_one({"_id": ObjectId(emotion_id)})
        if not emotion:
            raise ValueError(f"Emotion with ID {emotion_id} not found")
        emotion["_id"] = str(emotion["_id"])
        return emotion

    def get_all(self):
        emotions = self.collection.find()
        return [self._format(emotion) for emotion in emotions]

    def delete(self, emotion_id):
        result = self.collection.delete_one({"_id": ObjectId(emotion_id)})
        if result.deleted_count == 0:
            raise ValueError(f"Emotion with ID {emotion_id} not found")
        return {"message": "Emotion deleted successfully"}

    @staticmethod
    def _format(emotion):
        emotion["_id"] = str(emotion["_id"])
        return emotion
