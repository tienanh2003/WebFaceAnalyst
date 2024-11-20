from bson.objectid import ObjectId
from datetime import datetime, timezone

class AccountService:
    def __init__(self, db):
        self.collection = db.get_collection("accounts")

    def create(self, data):
        account = {
            "user_id": data["user_id"],
            "username": data["username"],
            "password": data["password"],
            "email": data["email"],
            "role": data["role"],
            "status": data["status"],
            "created_at": timezone.utcnow(),
            "updated_at": timezone.utcnow(),
        }
        result = self.collection.insert_one(account)
        return str(result.inserted_id)

    def get_by_id(self, account_id):
        account = self.collection.find_one({"_id": ObjectId(account_id)})
        if not account:
            raise ValueError(f"Account with ID {account_id} not found")
        account["_id"] = str(account["_id"])
        return account

    def get_all(self):
        accounts = self.collection.find()
        return [self._format(account) for account in accounts]

    def update(self, account_id, data):
        data["updated_at"] = datetime.utcnow()
        result = self.collection.update_one(
            {"_id": ObjectId(account_id)}, {"$set": data}
        )
        if result.matched_count == 0:
            raise ValueError(f"Account with ID {account_id} not found")
        return {"message": "Account updated successfully"}

    def delete(self, account_id):
        result = self.collection.delete_one({"_id": ObjectId(account_id)})
        if result.deleted_count == 0:
            raise ValueError(f"Account with ID {account_id} not found")
        return {"message": "Account deleted successfully"}

    @staticmethod
    def _format(account):
        account["_id"] = str(account["_id"])
        return account
