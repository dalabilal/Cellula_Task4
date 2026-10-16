import rag_DB

res = rag_DB.db.get(where={"source": "OOP.py"})
print("Found chunks:", len(res["ids"]))

if res["ids"]:
    rag_DB.db.delete(ids=res["ids"])

print("Total docs now:", rag_DB.db._collection.count())