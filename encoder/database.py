import logging
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def fetch_filtered_data(collection):
    return list(collection.find({"results": {"$exists": True}, "wrong": {"$exists": True}}))


def add_to_database():
    logging.info("Connecting to MongoDB...")
    client = MongoClient("mongodb://localhost:27017/")

    input_db = client["files"]
    output_db = client["database"]

    input_collections = [input_db["jsta"], input_db["stj"]]
    output_col = output_db["encoder"]

    logging.info("Fetching data from input collections...")
    files = [doc for col in input_collections for doc in fetch_filtered_data(col)]

    if not files:
        logging.warning("No valid data found in input collections. Skipping database update.")
        return

    logging.info(f"Deleting all documents from '{output_col.name}' collection...")
    output_col.delete_many({})

    logging.info(f"Inserting {len(files)} documents into '{output_col.name}' collection...")
    output_col.insert_many(files)

    logging.info("Database update completed successfully.")


def standardize_results():
    logging.info("Connecting to MongoDB...")
    client = MongoClient("mongodb://localhost:27017/")
    database = client["database"]
    collection = database["encoder"]

    logging.info(f"Fetching documents from '{collection.name}' collection...")
    files = list(collection.aggregate([]))

    if not files:
        logging.warning("No documents found in the collection. Skipping standardization.")
        return

    for file in files:
        result = file.get("results")

        if isinstance(result, list):
            updated_result = []
            modified = False

            for item in result:
                if isinstance(item, dict) and len(item) == 2:
                    keys = list(item.keys())

                    new_item = {
                        "pergunta": item.get(keys[0], "") if keys[0] != "pergunta" else item["pergunta"],
                        "resposta": item.get(keys[1], "") if keys[1] != "resposta" else item["resposta"],
                    }

                    updated_result.append(new_item)

                    if keys[0] != "pergunta" or keys[1] != "resposta":
                        modified = True
                else:
                    logging.warning(f"Skipping malformed entry in document {file['_id']}: {item}")
                    updated_result.append(item)

            if modified:
                collection.update_one(
                    {"_id": file["_id"]},
                    {"$set": {"results": updated_result}}
                )
                logging.info(f"Updated document {file['_id']} to enforce key standardization.")

        elif isinstance(result, dict):
            first_key = next(iter(result), None)
            if first_key and isinstance(result[first_key], list):
                new_value = result[first_key]

                collection.update_one(
                    {"_id": file["_id"]},
                    {"$set": {"results": new_value}}
                )
                logging.info(f"Reformatted 'results' field in document {file['_id']}.")

    logging.info("Standardization process completed.")


standardize_results()
