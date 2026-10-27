"""Merge every crawled collection in ``files`` into a single ``database.pool`` collection.

The target collection is cleared first. Each copied document gets an ``_origin_collection``
field with the name of the court collection it came from.
"""

import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from pymongo.errors import BulkWriteError, ConnectionFailure, OperationFailure
from tqdm import tqdm

from db import connect

SOURCE_DB = "files"
SOURCE_COLLECTIONS = [
    "cajp", "csm", "jcon", "jsta", "jstj", "jtca", "jtcampca", "jtcampct",
    "jtcn", "jtrc", "jtre", "jtrg", "jtrl", "jtrp", "pgr", "stj"
]
TARGET_DB = "database"
TARGET_COLLECTION = "pool"
ADD_ORIGIN_FIELD = True
BATCH_SIZE = 1000
MAX_RETRIES = 3
RETRY_DELAY = 1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@contextmanager
def get_mongo_client():
    client = None
    try:
        client = connect(serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        logger.info("Successfully connected to MongoDB")
        yield client
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


def batch_iterator(cursor, batch_size: int) -> Generator[list[dict[Any, Any]], None, None]:
    try:
        while True:
            batch = []
            for _ in range(batch_size):
                try:
                    doc = next(cursor)
                    batch.append(doc)
                except StopIteration:
                    break

            if not batch:
                break
            yield batch
    finally:
        cursor.close()


def clear_target_collection(target_collection) -> None:
    try:
        result = target_collection.delete_many({})
        logger.info(f"Target collection cleared. Deleted {result.deleted_count} documents.")
    except OperationFailure as e:
        logger.error(f"Failed to clear target collection: {e}")
        raise


def get_collection_stats(db, collection_name: str) -> dict[str, Any]:
    try:
        count = db[collection_name].count_documents({})
        return {"count": count, "exists": True}
    except OperationFailure as e:
        logger.warning(f"Collection {collection_name} might not exist or is inaccessible: {e}")
        return {"count": 0, "exists": False}


def insert_batch_with_retry(target_collection, batch: list[dict], collection_name: str) -> int:
    for attempt in range(MAX_RETRIES):
        try:
            result = target_collection.insert_many(batch, ordered=False)
            return len(result.inserted_ids)
        except BulkWriteError as e:
            inserted_count = e.details.get('nInserted', 0)
            write_errors = e.details.get('writeErrors', [])

            logger.warning(
                f"Bulk insert partial failure in {collection_name} "
                f"(attempt {attempt + 1}/{MAX_RETRIES}): "
                f"inserted {inserted_count}/{len(batch)}, "
                f"errors: {len(write_errors)}"
            )

            for error in write_errors[:3]:
                logger.warning(f"Write error: {error}")

            if attempt == MAX_RETRIES - 1:
                logger.error(f"Max retries reached for collection {collection_name}")
                return inserted_count

            time.sleep(RETRY_DELAY * (2 ** attempt))

        except OperationFailure as e:
            logger.error(f"Operation failed for collection {collection_name}: {e}")
            if attempt == MAX_RETRIES - 1:
                return 0
            time.sleep(RETRY_DELAY)

    return 0


def process_collection(source_db, target_collection, collection_name: str) -> int:
    logger.info(f"Starting processing of collection: {collection_name}")

    stats = get_collection_stats(source_db, collection_name)
    if not stats["exists"] or stats["count"] == 0:
        logger.warning(f"Collection {collection_name} is empty or doesn't exist, skipping")
        return 0

    total_count = stats["count"]
    processed_count = 0
    inserted_count = 0

    try:
        cursor = source_db[collection_name].find({}, no_cursor_timeout=True)

        with tqdm(total=total_count, unit="docs", desc=f"Processing {collection_name}") as pbar:
            for batch in batch_iterator(cursor, BATCH_SIZE):
                if ADD_ORIGIN_FIELD:
                    for doc in batch:
                        doc["_origin_collection"] = collection_name

                batch_inserted = insert_batch_with_retry(target_collection, batch, collection_name)
                inserted_count += batch_inserted
                processed_count += len(batch)

                pbar.update(len(batch))
                pbar.set_postfix({
                    'inserted': inserted_count,
                    'success_rate': f"{(inserted_count / processed_count) * 100:.1f}%"
                })

        logger.info(
            f"Completed {collection_name}: "
            f"processed {processed_count}, inserted {inserted_count}"
        )

    except Exception as e:
        logger.error(f"Unexpected error processing collection {collection_name}: {e}")
        return inserted_count

    return inserted_count


def validate_configuration():
    if not SOURCE_COLLECTIONS:
        raise ValueError("No source collections specified")

    if BATCH_SIZE <= 0:
        raise ValueError("Batch size must be positive")

    logger.info(f"Configuration validated: {len(SOURCE_COLLECTIONS)} collections, batch size {BATCH_SIZE}")


def main():
    start_time = time.time()
    logger.info("Starting MongoDB migration")

    try:
        validate_configuration()

        with get_mongo_client() as client:
            source_db = client[SOURCE_DB]
            target_db = client[TARGET_DB]
            target_collection = target_db[TARGET_COLLECTION]

            clear_target_collection(target_collection)

            total_inserted = 0
            successful_collections = 0

            for i, collection_name in enumerate(SOURCE_COLLECTIONS, 1):
                logger.info(f"Processing collection {i}/{len(SOURCE_COLLECTIONS)}: {collection_name}")

                try:
                    inserted = process_collection(source_db, target_collection, collection_name)
                    total_inserted += inserted
                    if inserted > 0:
                        successful_collections += 1

                except Exception as e:
                    logger.error(f"Failed to process collection {collection_name}: {e}")
                    continue

            elapsed_time = time.time() - start_time
            logger.info(
                f"Migration completed in {elapsed_time:.2f} seconds\n"
                f"✅ Successfully processed {successful_collections}/{len(SOURCE_COLLECTIONS)} collections\n"
                f"✅ Total documents inserted: {total_inserted:,}\n"
                f"✅ Average rate: {total_inserted / elapsed_time:.1f} docs/second"
            )

            if ADD_ORIGIN_FIELD and total_inserted > 0:
                try:
                    target_collection.create_index("_origin_collection")
                    logger.info("Created index on _origin_collection field")
                except Exception as e:
                    logger.warning(f"Failed to create index on _origin_collection: {e}")

    except Exception as e:
        logger.error(f"Migration failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
