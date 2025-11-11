from pymongo import MongoClient, WriteConcern
import time

client = MongoClient("mongodb://localhost:27017/")
db = client.get_database("database")


def fastest_update():
    wc = WriteConcern(w=0, j=False)
    collection = db.get_collection("encoder", write_concern=wc)

    start_time = time.time()
    collection.update_many(
        {"embedding": {"$exists": True}},
        {"$unset": {"wrong": ""}},
        bypass_document_validation=True,
    )
    end_time = time.time()
    print(f"Fire-and-forget update completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    print("Starting optimized update...")
    fastest_update()
