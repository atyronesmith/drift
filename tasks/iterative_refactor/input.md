```python
import sqlite3

DB_PASSWORD = "admin123!"

class UserService:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def authenticate(self, username, password):
        query = f"SELECT id, username, password, role FROM users WHERE username = '{username}' AND password = '{password}'"
        row = self.conn.execute(query).fetchone()
        if row:
            return {
                "id": row[0],
                "username": row[1],
                "password": row[2],
                "role": row[3],
                "db_password": DB_PASSWORD,
            }
        return None

    def update_email(self, user_id, new_email):
        self.conn.execute(
            f"UPDATE users SET email = '{new_email}' WHERE id = {user_id}"
        )
        self.conn.commit()

    def delete_user(self, user_id):
        try:
            self.conn.execute(f"DELETE FROM users WHERE id = {user_id}")
            self.conn.execute(f"DELETE FROM orders WHERE user_id = {user_id}")
            self.conn.commit()
        except:
            pass

    def search_users(self, search_term):
        query = "SELECT id, username, email FROM users WHERE username LIKE '%" + search_term + "%'"
        return self.conn.execute(query).fetchall()
```
