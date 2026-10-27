"""Remove the ``wrong`` field from every ``database.encoder`` document that has an embedding.

Run before ``helpers/embed.py`` to recompute ``wrong`` from the stored embeddings.
"""

import time

from pymongo import WriteConcern

from db import connect


def fastest_update():
    wc = WriteConcern(w=0, j=False)
    collection = connect().get_database("database").get_collection("encoder", write_concern=wc)

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
