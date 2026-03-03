from functools import wraps
from flask import abort, request
from flask_login import current_user
from web_app.services import get_db


def require_role(roles):
    """
    Decorator that checks if the current user has one of the specified roles.
    Roles list: ['CLIENTE', 'GESTOR', 'DIRECCION', 'ADMIN']
    """
    if isinstance(roles, str):
        roles = [roles]
        
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(401)  # Unauthorized
            
            if current_user.role not in roles and current_user.role != 'ADMIN':
                db = get_db()
                db.log_audit(current_user.id, 'access_denied_role', request.endpoint, None, {'required_roles': roles, 'user_role': current_user.role})
                abort(403)  # Forbidden
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def hotel_scoped(hotel_id_key='hotel_id'):
    """
    Decorator that ensures the hotel_id (from request or kwargs) 
    is within the current user's hotel_scope.
    Uses parameterized queries to prevent SQL injection.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(401)
            
            # Admins bypass scoping but we log access
            if current_user.role == 'ADMIN':
                return f(*args, **kwargs)
            
            # Get hotel_id from arguments
            val = kwargs.get(hotel_id_key)
            if val is None:
                return f(*args, **kwargs)

            db = get_db()
            
            # Check if we're resolving from document
            if hotel_id_key == 'doc_id':
                # Use parameterized query to prevent SQL injection
                resolved_hotel_id = None
                with db.get_connection() as conn:
                    cursor = db.get_cursor(conn)
                    ph = getattr(db, "placeholder", "%s")
                    cursor.execute(f"SELECT hotel_id FROM documents WHERE id = {ph}", (val,))
                    row = cursor.fetchone()
                    if row:
                        resolved_hotel_id = row[0] if isinstance(row, (tuple, list)) else row['hotel_id']

                # Fail closed: unscoped (NULL) documents are forbidden for non-admin users.
                if resolved_hotel_id is None:
                    db.log_audit(
                        current_user.id,
                        'access_denied_scope',
                        'document',
                        str(val),
                        {'endpoint': request.endpoint, 'reason': 'missing_hotel_id'},
                    )
                    abort(403)
                
                if str(resolved_hotel_id) not in [str(h) for h in current_user.hotel_scope]:
                    db.log_audit(current_user.id, 'access_denied_scope', 'document', str(val), {'endpoint': request.endpoint})
                    abort(403)
            else:
                hotel_id = val
                if hotel_id is not None and str(hotel_id) not in [str(h) for h in current_user.hotel_scope]:
                    db.log_audit(current_user.id, 'access_denied_scope', 'hotel', str(hotel_id), {'endpoint': request.endpoint})
                    abort(403)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def owner_scoped(doc_id_key: str = "doc_id"):
    """
    Decorator that ensures CLIENT users can only access their own documents (owner_id == current_user.id).

    Intended to be used alongside @hotel_scoped('doc_id') for full multi-tenant isolation.
    Uses parameterized queries to prevent SQL injection.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(401)

            role = str(getattr(current_user, "role", "")).upper()
            if role not in {"CLIENTE", "CLIENT"}:
                return f(*args, **kwargs)

            doc_id = kwargs.get(doc_id_key)
            if doc_id is None:
                return f(*args, **kwargs)

            db = get_db()
            
            # Use parameterized query to prevent SQL injection
            with db.get_connection() as conn:
                cursor = db.get_cursor(conn)
                ph = getattr(db, "placeholder", "%s")
                cursor.execute(f"SELECT owner_id FROM documents WHERE id = {ph}", (doc_id,))
                row = cursor.fetchone()

            # If the doc does not exist, let the handler return 404 (do not mask).
            if not row:
                return f(*args, **kwargs)

            owner_id = row[0] if isinstance(row, (tuple, list)) else row["owner_id"]
            if owner_id is None or str(owner_id) != str(current_user.id):
                db.log_audit(
                    current_user.id,
                    "access_denied_owner",
                    "document",
                    str(doc_id),
                    {"endpoint": request.endpoint},
                )
                abort(403)

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def financial_access_required():
    """
    Checks if the user has access to financial data.
    DIRECCION and ADMIN only.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(401)
            
            if current_user.role not in ['DIRECCION', 'ADMIN']:
                db = get_db()
                db.log_audit(current_user.id, 'access_denied_financial', 'financial_data', None, {'endpoint': request.endpoint})
                abort(403)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator
