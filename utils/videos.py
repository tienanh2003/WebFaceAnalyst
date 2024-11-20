from bson.objectid import ObjectId
from datetime import datetime, timezone


class VideoService:
    def __init__(self, db):
        self.collection = db.get_collection("videos")

    def create(self, data):
        video = {
            "video_name": data["video_name"],
            "upload_time": data["upload_time"],
            "processed": data["processed"],
            "created_at": timezone.utcnow(),
        }
        result = self.collection.insert_one(video)
        return str(result.inserted_id)

    def get_by_id(self, video_id):
        video = self.collection.find_one({"_id": ObjectId(video_id)})
        if not video:
            raise ValueError(f"Video with ID {video_id} not found")
        video["_id"] = str(video["_id"])
        return video

    def get_all(self):
        videos = self.collection.find()
        return [self._format(video) for video in videos]

    def delete(self, video_id):
        result = self.collection.delete_one({"_id": ObjectId(video_id)})
        if result.deleted_count == 0:
            raise ValueError(f"Video with ID {video_id} not found")
        return {"message": "Video deleted successfully"}

    @staticmethod
    def _format(video):
        video["_id"] = str(video["_id"])
        return video
