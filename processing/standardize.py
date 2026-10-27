"""Reduce every document in ``database.pool`` to ``_id`` and a single ``Texto Integral`` field.

Sources name the full-text field differently (for example "Texto Integral", "Decisão Integral" or
"Decisão Texto Integral"). Every field whose name contains "integral" (case-insensitive) is a
candidate. When there are several, the longest string wins. Documents with no such field are dropped.
"""

import time

from db import connect

DB_NAME = "database"
COLLECTION_NAME = "pool"


def main():
    client = connect()
    collection = client[DB_NAME][COLLECTION_NAME]

    count = collection.count_documents({})
    print(f"Standardizing collection: {DB_NAME}.{COLLECTION_NAME}")
    print(f"Total documents: {count:,}")

    start_time = time.time()

    pipeline = [
        {
            "$addFields": {
                "texto_fields": {
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
        },
        {
            "$match": {
                "texto_fields": {"$ne": []}
            }
        },
        {
            "$replaceRoot": {
                "newRoot": {
                    "_id": "$_id",
                    "Texto Integral": {
                        "$cond": {
                            "if": {"$eq": [{"$size": "$texto_fields"}, 1]},
                            "then": {"$arrayElemAt": ["$texto_fields.v", 0]},
                            "else": {
                                "$let": {
                                    "vars": {
                                        "max_field": {
                                            "$reduce": {
                                                "input": "$texto_fields",
                                                "initialValue": {"v": "", "len": 0},
                                                "in": {
                                                    "$cond": {
                                                        "if": {
                                                            "$gt": [
                                                                {
                                                                    "$cond": {
                                                                        "if": {"$eq": [{"$type": "$$this.v"}, "string"]},
                                                                        "then": {"$strLenCP": "$$this.v"},
                                                                        "else": 0
                                                                    }
                                                                },
                                                                "$$value.len"
                                                            ]
                                                        },
                                                        "then": {
                                                            "v": "$$this.v",
                                                            "len": {
                                                                "$cond": {
                                                                    "if": {"$eq": [{"$type": "$$this.v"}, "string"]},
                                                                    "then": {"$strLenCP": "$$this.v"},
                                                                    "else": 0
                                                                }
                                                            }
                                                        },
                                                        "else": "$$value"
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    "in": "$$max_field.v"
                                }
                            }
                        }
                    }
                }
            }
        },
        {"$out": COLLECTION_NAME}
    ]

    print("🚀 Starting aggregation pipeline...")

    try:
        collection.aggregate(pipeline, allowDiskUse=True)

        end_time = time.time()
        elapsed_time = end_time - start_time

        new_count = collection.count_documents({})
        updated = new_count
        deleted = count - new_count

        print(f"✅ Done! Processed {count:,} documents in {elapsed_time:.2f} seconds")
        print(f"⚡ Performance: {count / elapsed_time:.0f} documents/second")
        print(f"📊 Updated: {updated:,}, Deleted: {deleted:,}")

        # Sample a document to show the result
        sample_doc = collection.find_one({})
        if sample_doc:
            print("📝 Sample standardized document:")
            print(f"   _id: {sample_doc['_id']}")
            texto_preview = sample_doc.get('Texto Integral', '')
            if isinstance(texto_preview, str) and len(texto_preview) > 100:
                texto_preview = texto_preview[:100] + "..."
            print(f"   Texto Integral: {texto_preview}")

    finally:
        client.close()
        print("🔒 Database connection closed")


if __name__ == "__main__":
    main()
