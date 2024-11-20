from bson.objectid import ObjectId
from datetime import datetime


class FaceEmbeddingService:
    def __init__(self, db):
        self.collection = db.get_collection("face_embeddings")

    def create(self, data):
        embedding = {
            "user_id": data.get("user_id"),
            "embedding_vector": data["embedding_vector"],
            "created_at": datetime.utcnow(),
        }
        result = self.collection.insert_one(embedding)
        return str(result.inserted_id)

    def get_by_id(self, embedding_id):
        embedding = self.collection.find_one({"_id": ObjectId(embedding_id)})
        if not embedding:
            raise ValueError(f"Embedding with ID {embedding_id} not found")
        embedding["_id"] = str(embedding["_id"])
        return embedding

    def get_all(self):
        embeddings = self.collection.find()
        return [self._format(embedding) for embedding in embeddings]

    def delete(self, embedding_id):
        result = self.collection.delete_one({"_id": ObjectId(embedding_id)})
        if result.deleted_count == 0:
            raise ValueError(f"Embedding with ID {embedding_id} not found")
        return {"message": "Embedding deleted successfully"}

    @staticmethod
    def _format(embedding):
        embedding["_id"] = str(embedding["_id"])
        return embedding
