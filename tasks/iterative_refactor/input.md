```python
import sqlite3
import threading
import time

DB_PASSWORD = "admin123!"
API_SECRET = "sk-prod-9f8e7d6c5b4a3210"

_cache = {}

class UserService:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.lock = threading.Lock()

    def authenticate(self, username, password):
        query = f"SELECT id, username, password, role, email FROM users WHERE username = '{username}' AND password = '{password}'"
        row = self.conn.execute(query).fetchone()
        if row:
            return {
                "id": row[0],
                "username": row[1],
                "password": row[2],
                "role": row[3],
                "email": row[4],
                "db_password": DB_PASSWORD,
            }
        return None

    def get_user_orders(self, user_id):
        orders = self.conn.execute(
            f"SELECT * FROM orders WHERE user_id = {user_id}"
        ).fetchall()
        result = []
        for order in orders:
            items = self.conn.execute(
                f"SELECT * FROM order_items WHERE order_id = {order[0]}"
            ).fetchall()
            product_names = []
            for item in items:
                product = self.conn.execute(
                    f"SELECT name FROM products WHERE id = {item[2]}"
                ).fetchone()
                product_names.append(product[0])
            result.append({
                "order_id": order[0],
                "total": order[2],
                "items": product_names,
            })
        return result

    def update_user_email(self, user_id, new_email):
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

    def get_all_users(self):
        rows = self.conn.execute("SELECT * FROM users").fetchall()
        users = []
        for row in rows:
            users.append({
                "id": row[0],
                "username": row[1],
                "password": row[2],
                "role": row[3],
                "email": row[4],
            })
        return users

    def bulk_update_prices(self, price_multiplier):
        products = self.conn.execute("SELECT id, price FROM products").fetchall()
        for product in products:
            new_price = round(product[1] * price_multiplier, 2)
            self.conn.execute(
                f"UPDATE products SET price = {new_price} WHERE id = {product[0]}"
            )
        self.conn.commit()

    def search_users(self, search_term):
        query = "SELECT * FROM users WHERE username LIKE '%" + search_term + "%' OR email LIKE '%" + search_term + "%'"
        return self.conn.execute(query).fetchall()

    def get_user_with_cache(self, user_id):
        if user_id in _cache:
            return _cache[user_id]
        row = self.conn.execute(
            f"SELECT * FROM users WHERE id = {user_id}"
        ).fetchone()
        if row:
            user = {"id": row[0], "username": row[1], "password": row[2]}
            _cache[user_id] = user
            return user
        return None

    def transfer_orders(self, from_user_id, to_user_id):
        self.conn.execute(
            f"UPDATE orders SET user_id = {to_user_id} WHERE user_id = {from_user_id}"
        )
        self.conn.commit()
```
