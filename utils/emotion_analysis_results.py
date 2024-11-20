from bson.objectid import ObjectId
from datetime import datetime, timezone


class EmotionAnalysisResultService:
    def __init__(self, db):
        self.collection = db.get_collection("emotion_analysis_results")

    def create(self, data):
        result = {
            "session_id": data["session_id"],
            "user_id": data.get("user_id"),
            "emotion_id": data["emotion_id"],
            "detected_at": data["detected_at"],
            "confidence": data["confidence"],
            "created_at": timezone.utcnow(),
        }
        result = self.collection.insert_one(result)
        return str(result.inserted_id)

    def get_by_id(self, result_id):
        result = self.collection.find_one({"_id": ObjectId(result_id)})
        if not result:
            raise ValueError(f"EmotionAnalysisResult with ID {result_id} not found")
        result["_id"] = str(result["_id"])
        return result

    def get_all(self):
        results = self.collection.find()
        return [self._format(result) for result in results]

    def delete(self, result_id):
        result = self.collection.delete_one({"_id": ObjectId(result_id)})
        if result.deleted_count == 0:
            raise ValueError(f"EmotionAnalysisResult with ID {result_id} not found")
        return {"message": "Result deleted successfully"}

    @staticmethod
    def _format(result):
        result["_id"] = str(result["_id"])
        return result
