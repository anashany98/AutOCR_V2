import logging
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

logger = logging.getLogger(__name__)

class User(UserMixin):
    ROLES = ['CLIENTE', 'GESTOR', 'DIRECCION', 'ADMIN']

    def __init__(self, id, username, role, client_id=None, hotel_scope=None):
        self.id = str(id)
        self.username = username
        self.role = role.upper() if role else 'CLIENTE'
        if self.role not in self.ROLES:
             self.role = 'CLIENTE'
        self.client_id = client_id
        # hotel_scope is a list of IDs or None (all if ADMIN)
        import json
        try:
            self.hotel_scope = json.loads(hotel_scope) if isinstance(hotel_scope, str) else (hotel_scope or [])
        except:
            self.hotel_scope = []

    @property
    def is_admin(self):
        return self.role == 'ADMIN'

class AuthManager:
    def __init__(self, db_manager):
        self.db = db_manager

    def get_user(self, user_id):
        query = "SELECT id, username, role, client_id, hotel_scope FROM users WHERE id = ?"
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query.replace('?', self.db.placeholder), (user_id,))
            row = cursor.fetchone()
            if row:
                return User(row[0], row[1], row[2], row[3], row[4])
        return None

    def get_user_by_username(self, username):
        query = "SELECT id, username, role, client_id, password_hash, hotel_scope FROM users WHERE username = ?"
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query.replace('?', self.db.placeholder), (username,))
            row = cursor.fetchone()
            if row:
                user = User(row[0], row[1], row[2], row[3], row[5])
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

    def list_users(self):
        """List all users in the system."""
        query = "SELECT id, username, role, client_id, hotel_scope FROM users ORDER BY username ASC"
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return [User(row[0], row[1], row[2], row[3], row[4]) for row in rows]

    def update_user_role(self, user_id, role):
        """Update a user's role."""
        if role.upper() not in User.ROLES:
            return False, "Rol inválido"
        
        query = "UPDATE users SET role = ? WHERE id = ?"
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query.replace('?', self.db.placeholder), (role.upper(), user_id))
                conn.commit()
            return True, "Rol actualizado"
        except Exception as e:
            return False, str(e)

    def update_user_hotel_scope(self, user_id, hotel_scope):
        """Update a user's hotel access scope."""
        import json
        scope_json = json.dumps(hotel_scope) if isinstance(hotel_scope, list) else hotel_scope
        
        query = "UPDATE users SET hotel_scope = ? WHERE id = ?"
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query.replace('?', self.db.placeholder), (scope_json, user_id))
                conn.commit()
            return True, "Alcance de hoteles actualizado"
        except Exception as e:
            return False, str(e)
