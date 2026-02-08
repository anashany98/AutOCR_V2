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
                abort(401) # Unauthorized
            
            if current_user.role not in roles and current_user.role != 'ADMIN':
                db = get_db()
                db.log_audit(current_user.id, 'access_denied_role', request.endpoint, None, {'required_roles': roles, 'user_role': current_user.role})
                abort(403) # Forbidden
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def hotel_scoped(hotel_id_key='hotel_id'):
    """
    Decorator that ensures the hotel_id (from request or kwargs) 
    is within the current user's hotel_scope.
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

            hotel_id = None
            db = get_db()
            if hotel_id_key == 'doc_id':
                # Resolve hotel_id from document
                with db.get_connection() as conn:
                    cursor = db.get_cursor(conn)
                    cursor.execute(f"SELECT hotel_id FROM documents WHERE id = {db.placeholder}", (val,))
                    row = cursor.fetchone()
                    if row:
                        hotel_id = row[0] if isinstance(row, (tuple, list)) else row['hotel_id']
            else:
                hotel_id = val

            if hotel_id and str(hotel_id) not in [str(h) for h in current_user.hotel_scope]:
                db.log_audit(current_user.id, 'access_denied_scope', 'hotel', str(hotel_id), {'endpoint': request.endpoint})
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
