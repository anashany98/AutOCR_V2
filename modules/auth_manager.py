import json
import logging
from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)


class User(UserMixin):
    ROLES = ["CLIENTE", "GESTOR", "DIRECCION", "ADMIN"]

    def __init__(self, id, username, role, preferences=None, is_verified=True):
        self.id = str(id)
        self.username = username
        self.role = role.upper() if isinstance(role, str) and role else "CLIENTE"
        if self.role not in self.ROLES:
            self.role = "CLIENTE"
        self.is_verified = bool(is_verified)

        # hotel_scope may come from:
        # - preferences JSON object: {"hotel_scope": [...]}
        # - hotel_scope JSON list: [...]
        try:
            parsed = json.loads(preferences) if isinstance(preferences, str) else preferences
            if isinstance(parsed, dict):
                scope = parsed.get("hotel_scope", [])
            elif isinstance(parsed, list):
                scope = parsed
            else:
                scope = []
            self.hotel_scope = [str(v) for v in scope if v is not None]
        except Exception:
            self.hotel_scope = []

    @property
    def is_admin(self):
        return self.role == "ADMIN"


class AuthManager:
    def __init__(self, db_manager):
        self.db = db_manager
        self.last_error = None
        self._scope_column_cache = None

    def _is_postgres(self) -> bool:
        return str(getattr(self.db, "engine_type", "")).lower() == "postgresql"

    def _id_expr(self) -> str:
        return f"{self.db.placeholder}::uuid" if self._is_postgres() else self.db.placeholder

    def _scope_column(self) -> str:
        if self._scope_column_cache:
            return self._scope_column_cache

        for candidate in ("preferences", "hotel_scope"):
            try:
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT {candidate} FROM users LIMIT 1")
                self._scope_column_cache = candidate
                return candidate
            except Exception:
                continue

        # Safe local fallback.
        self._scope_column_cache = "hotel_scope"
        return self._scope_column_cache

    def get_user(self, user_id):
        scope_col = self._scope_column()
        query = (
            "SELECT id, username, role, "
            + scope_col
            + ", COALESCE(is_verified, 1) FROM users WHERE id = "
            + self.db.placeholder
        )
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (str(user_id),))
            row = cursor.fetchone()
            if row:
                return User(row[0], row[1], row[2], row[3], is_verified=row[4])
        return None

    def get_user_by_username(self, username):
        scope_col = self._scope_column()
        query = (
            "SELECT id, username, role, password_hash, "
            + scope_col
            + ", COALESCE(is_verified, 1) FROM users WHERE username = "
            + self.db.placeholder
        )
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (username,))
            row = cursor.fetchone()
            if row:
                user = User(row[0], row[1], row[2], row[4], is_verified=row[5])
                user.password_hash = row[3]
                return user
        return None

    def create_user(self, username, password, role="CLIENTE"):
        existing = self.get_user_by_username(username)
        if existing:
            return False, "Usuario ya existe"

        password_hash = generate_password_hash(password)

        ph = self.db.placeholder
        query = f"INSERT INTO users (username, password_hash, role) VALUES ({ph}, {ph}, {ph})"
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (username, password_hash, role))
                conn.commit()
            return True, "Usuario creado exitosamente"
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, str(e)

    def verify_password(self, user, password):
        if not user or not hasattr(user, "password_hash"):
            return False
        return check_password_hash(user.password_hash, password)

    def verify_login(self, username, password):
        self.last_error = "invalid_credentials"
        user = self.get_user_by_username(username)
        if user and self.verify_password(user, password):
            # SECURITY: Email verification check
            # TODO: Re-enable email verification before production deployment
            # Uncomment the following lines when ready:
            # if not user.is_verified:
            #     self.last_error = "email_not_verified"
            #     return None
            self.last_error = None
            return user
        return None

    def list_users(self):
        """List all users in the system."""
        scope_col = self._scope_column()
        query = (
            "SELECT id, username, role, "
            + scope_col
            + ", COALESCE(is_verified, 1) FROM users ORDER BY username ASC"
        )
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return [User(row[0], row[1], row[2], row[3], is_verified=row[4]) for row in rows]

    def update_user_role(self, user_id, role):
        """Update a user's role."""
        if role.upper() not in User.ROLES:
            return False, "Rol invalido"

        ph = self.db.placeholder
        query = f"UPDATE users SET role = {ph} WHERE id = {self._id_expr()}"
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (role.upper(), str(user_id)))
                conn.commit()
            return True, "Rol actualizado"
        except Exception as e:
            return False, str(e)

    def update_user_hotel_scope(self, user_id, hotel_scope):
        """Update a user's hotel access scope."""
        scope_values = hotel_scope if isinstance(hotel_scope, list) else []
        scope_json = json.dumps({"hotel_scope": scope_values})
        scope_col = self._scope_column()

        ph = self.db.placeholder
        if scope_col == "preferences":
            value_expr = f"{ph}::jsonb" if self._is_postgres() else ph
            query = f"UPDATE users SET preferences = {value_expr} WHERE id = {self._id_expr()}"
            value = scope_json
        else:
            query = f"UPDATE users SET hotel_scope = {ph} WHERE id = {self._id_expr()}"
            value = json.dumps(scope_values)

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (value, str(user_id)))
                conn.commit()
            return True, "Alcance de hoteles actualizado"
        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------------------ #
    # Email Verification & Password Reset
    # ------------------------------------------------------------------ #

    def create_user_with_email(self, username, email, password, role="client", allow_elevated_role=False):
        existing = self.get_user_by_username(username)
        if existing:
            return False, "Usuario ya existe"

        # Check email too
        ph = self.db.placeholder
        query = f"SELECT id FROM users WHERE email = {ph}"
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (email,))
            if cursor.fetchone():
                return False, "Email ya registrado"

        password_hash = generate_password_hash(password)

        # Generate Verification Token
        import secrets

        token = secrets.token_urlsafe(32)

        # Public registration must never elevate privileges.
        final_role = "CLIENTE"
        role_raw = str(role or "").strip()
        role_upper = role_raw.upper()
        if allow_elevated_role:
            if role_raw.lower() == "personal":
                final_role = "GESTOR"
            elif role_upper in User.ROLES:
                final_role = role_upper

        query = f"""
            INSERT INTO users (username, email, password_hash, role, is_verified, verification_token)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (username, email, password_hash, final_role, False, token))
                conn.commit()
            return True, token  # Return token to send email
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, str(e)

    def verify_email(self, token):
        ph = self.db.placeholder
        query = f"SELECT id FROM users WHERE verification_token = {ph}"
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (token,))
                row = cursor.fetchone()
                if not row:
                    return False, "Token invalido"

                user_id = row[0]
                upd = f"UPDATE users SET is_verified = TRUE, verification_token = NULL WHERE id = {ph}"
                cursor.execute(upd, (user_id,))
                conn.commit()
            return True, "Email verificado correctamente"
        except Exception as e:
            return False, str(e)

    def request_password_reset(self, email):
        ph = self.db.placeholder
        query = f"SELECT id FROM users WHERE email = {ph}"
        import secrets
        from datetime import timedelta

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (email,))
                row = cursor.fetchone()
                if not row:
                    return False, "Email no encontrado", None

                user_id = row[0]
                token = secrets.token_urlsafe(32)
                expiry = datetime.now() + timedelta(hours=1)

                upd = f"UPDATE users SET reset_token = {ph}, token_expiry = {ph} WHERE id = {ph}"
                cursor.execute(upd, (token, expiry, user_id))
                conn.commit()
                return True, "Token generado", token
        except Exception as e:
            return False, str(e), None

    def reset_password(self, token, new_password):
        ph = self.db.placeholder
        query = f"SELECT id, token_expiry FROM users WHERE reset_token = {ph}"
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (token,))
                row = cursor.fetchone()
                if not row:
                    return False, "Token invalido"

                user_id = row[0]
                expiry = row[1]

                # token_expiry is already a datetime from PostgreSQL
                if isinstance(expiry, str):
                    expiry = datetime.fromisoformat(expiry)
                if datetime.now(expiry.tzinfo if expiry.tzinfo else None) > expiry:
                    return False, "Token expirado"

                new_hash = generate_password_hash(new_password)
                upd = f"UPDATE users SET password_hash = {ph}, reset_token = NULL, token_expiry = NULL WHERE id = {ph}"
                cursor.execute(upd, (new_hash, user_id))
                conn.commit()
            return True, "Contrasena restablecida"
        except Exception as e:
            return False, str(e)
