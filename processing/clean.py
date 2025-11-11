from pymongo import MongoClient
import time

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "database"
COLLECTION_NAME = "pool"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

count = collection.count_documents({})
print(f"Cleaning collection: {DB_NAME}.{COLLECTION_NAME}")
print(f"Total documents: {count:,}")

start_time = time.time()

pipeline = [
    {
        "$replaceRoot": {
            "newRoot": {
                "$mergeObjects": [
                    {"_id": "$_id"},
                    {
                        "$arrayToObject": {
                            "$filter": {
                                "input": {"$objectToArray": "$$ROOT"},
                                "cond": {
                                    "$and": [
                                        {"$ne": ["$$this.k", "_id"]},
                                        {
                                            "$regexMatch": {
                                                "input": "$$this.k",
                                                "regex": "integral",
                                                "options": "i"
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    }
                ]
            }
        }
    },
    {"$out": COLLECTION_NAME}
]

print("🚀 Starting aggregation pipeline...")

try:
    result = list(collection.aggregate(pipeline, allowDiskUse=True))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"✅ Done! Cleaned {count:,} documents in {elapsed_time:.2f} seconds")
    print(f"⚡ Performance: {count / elapsed_time:.0f} documents/second")

    new_count = collection.count_documents({})
    print(f"📊 Documents after cleaning: {new_count:,}")

    if new_count != count:
        print(f"⚠️  Warning: Document count changed from {count:,} to {new_count:,}")

    sample_doc = collection.find_one({})
    if sample_doc:
        print(f"📝 Sample cleaned document fields: {list(sample_doc.keys())}")

except Exception as e:
    print(f"❌ Error during aggregation: {e}")

finally:
    client.close()
    print("🔒 Database connection closed")
