import logging
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

logger = logging.getLogger(__name__)

class User(UserMixin):
    def __init__(self, id, username, role, client_id=None):
        self.id = str(id)
        self.username = username
        self.role = role
        self.client_id = client_id

    @property
    def is_admin(self):
        return self.role == 'admin'

class AuthManager:
    def __init__(self, db_manager):
        self.db = db_manager

    def get_user(self, user_id):
        query = "SELECT id, username, role, client_id FROM users WHERE id = ?"
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query.replace('?', self.db.placeholder), (user_id,))
            row = cursor.fetchone()
            if row:
                return User(row[0], row[1], row[2], row[3])
        return None

    def get_user_by_username(self, username):
        query = "SELECT id, username, role, client_id, password_hash FROM users WHERE username = ?"
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query.replace('?', self.db.placeholder), (username,))
            row = cursor.fetchone()
            if row:
                user = User(row[0], row[1], row[2], row[3])
                user.password_hash = row[4]
                return user
        return None

    def create_user(self, username, password, role='client', client_id=None):
        existing = self.get_user_by_username(username)
        if existing:
            return False, "Usuario ya existe"

        password_hash = generate_password_hash(password)
        created_at = datetime.now().isoformat()
        
        query = """
            INSERT INTO users (username, password_hash, role, client_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query.replace('?', self.db.placeholder), 
                             (username, password_hash, role, client_id, created_at))
                conn.commit()
            return True, "Usuario creado exitosamente"
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, str(e)

    def verify_password(self, user, password):
        if not user or not hasattr(user, 'password_hash'):
            return False
        return check_password_hash(user.password_hash, password)

    def verify_login(self, username, password):
        user = self.get_user_by_username(username)
        if user and self.verify_password(user, password):
            return user
        return None
