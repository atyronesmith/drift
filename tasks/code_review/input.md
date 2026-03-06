```python
def process_users(db, user_ids):
    results = []
    for uid in user_ids:
        query = f"SELECT * FROM users WHERE id = {uid}"
        row = db.execute(query).fetchone()
        if row:
            user = {"id": row[0], "name": row[1], "email": row[2]}
            user["password"] = row[3]
            results.append(user)
    return results
```
