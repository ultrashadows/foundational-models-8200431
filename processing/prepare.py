from pymongo import MongoClient
import time

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "database"
COLLECTION_NAME = "pool"
MIN_WORDS = 50

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

initial_count = collection.count_documents({})
print(f"Filtering collection: {DB_NAME}.{COLLECTION_NAME}")
print(f"Initial documents: {initial_count:,}")
print(f"Minimum words required: {MIN_WORDS}")

start_time = time.time()

pipeline = [
    {
        "$addFields": {
            "word_count": {
                "$cond": {
                    "if": {
                        "$and": [
                            {"$ne": ["$Texto Integral", None]},
                            {"$eq": [{"$type": "$Texto Integral"}, "string"]}
                        ]
                    },
                    "then": {
                        "$size": {
                            "$filter": {
                                "input": {
                                    "$split": [
                                        {
                                            "$trim": {
                                                "input": "$Texto Integral"
                                            }
                                        },
                                        " "
                                    ]
                                },
                                "cond": {"$ne": ["$$this", ""]}
                            }
                        }
                    },
                    "else": 0
                }
            }
        }
    },
    {
        "$match": {
            "word_count": {"$gte": MIN_WORDS}
        }
    },
    {
        "$project": {
            "word_count": 0
        }
    },
    # Stage 4: Replace the collection
    {"$out": COLLECTION_NAME}
]

print("🚀 Starting word count filtering...")

try:
    # Execute the aggregation pipeline
    result = list(collection.aggregate(pipeline, allowDiskUse=True))

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Get final count
    final_count = collection.count_documents({})
    deleted_count = initial_count - final_count

    print(f"✅ Done! Filtered {initial_count:,} documents in {elapsed_time:.2f} seconds")
    print(f"⚡ Performance: {initial_count / elapsed_time:.0f} documents/second")
    print(f"📊 Kept: {final_count:,} documents")
    print(f"🗑️  Deleted: {deleted_count:,} documents (under {MIN_WORDS} words)")
    print(f"📈 Retention rate: {(final_count / initial_count) * 100:.1f}%")

    # Show some statistics
    if final_count > 0:
        # Sample a few documents to show word counts
        sample_docs = list(collection.aggregate([
            {
                "$addFields": {
                    "word_count": {
                        "$size": {
                            "$filter": {
                                "input": {"$split": [{"$trim": {"input": "$Texto Integral"}}, " "]},
                                "cond": {"$ne": ["$$this", ""]}
                            }
                        }
                    }
                }
            },
            {"$sample": {"size": 3}},
            {"$project": {"_id": 1, "word_count": 1}}
        ]))

        print(f"📝 Sample word counts from remaining documents:")
        for doc in sample_docs:
            print(f"   Document {doc['_id']}: {doc['word_count']} words")

except Exception as e:
    print(f"❌ Error during filtering: {e}")

finally:
    client.close()
    print("🔒 Database connection closed")