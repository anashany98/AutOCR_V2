from flask import Blueprint, render_template, redirect, url_for, flash, request, abort
from flask_login import login_required, current_user
from web_app.services import get_db
from modules.auth_manager import AuthManager, User
from web_app.security.security_decorators import require_role
import json

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

def get_auth_manager():
    return AuthManager(get_db())

@admin_bp.before_request
@login_required
@require_role('ADMIN')
def before_request():
    """Ensure only admins can access these routes."""
    pass

@admin_bp.route('/hotels', methods=['GET', 'POST'])
def manage_hotels():
    db = get_db()
    
    if request.method == 'POST':
        action = request.form.get('action')
        name = request.form.get('name')
        code = request.form.get('code')
        description = request.form.get('description', '')
        
        if action == 'create':
            db.create_hotel(name, code, description)
            db.log_audit(current_user.id, 'create_hotel', 'hotel', None, {'name': name, 'code': code})
            flash(f'Hotel {name} creado exitosamente', 'success')
        elif action == 'update':
            hotel_id = request.form.get('hotel_id')
            db.update_hotel(hotel_id, name, code, description)
            db.log_audit(current_user.id, 'update_hotel', 'hotel', hotel_id, {'name': name})
            flash(f'Hotel {name} actualizado', 'success')
            
        return redirect(url_for('admin.manage_hotels'))

    hotels = db.get_hotels()
    return render_template('admin/hotels.html', hotels=hotels)

@admin_bp.route('/users', methods=['GET', 'POST'])
def manage_users():
    auth = get_auth_manager()
    db = get_db()
    
    if request.method == 'POST':
        action = request.form.get('action')
        user_id = request.form.get('user_id')
        
        if action == 'update_role':
            role = request.form.get('role')
            success, msg = auth.update_user_role(user_id, role)
            if success:
                db.log_audit(current_user.id, 'update_user_role', 'user', user_id, {'role': role})
                flash(msg, 'success')
            else:
                flash(msg, 'danger')
                
        elif action == 'update_scope':
            hotels_list = request.form.getlist('hotels')
            # hotel_scope is stored as JSON string in DB, update_user_hotel_scope handles serialization
            success, msg = auth.update_user_hotel_scope(user_id, hotels_list)
            if success:
                db.log_audit(current_user.id, 'update_user_scope', 'user', user_id, {'scope': hotels_list})
                flash(msg, 'success')
            else:
                flash(msg, 'danger')
                
        return redirect(url_for('admin.manage_users'))

    users = auth.list_users()
    hotels = db.get_hotels()
    return render_template('admin/users.html', users=users, hotels=hotels, roles=User.ROLES)

@admin_bp.route('/audit')
def view_audit():
    db = get_db()
    # Need to add get_audit_logs to DBManager
    query = """
        SELECT a.id, a.user_id, u.username, a.action, a.resource_type, a.resource_id, a.details, a.timestamp
        FROM audit_logs a
        LEFT JOIN users u ON a.user_id = u.id
        ORDER BY a.timestamp DESC LIMIT 500
    """
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        cursor.execute(query)
        logs = cursor.fetchall()
        
    return render_template('admin/audit.html', logs=logs)

@admin_bp.route('/queue')
def monitor_queue():
    """
    Monitor Celery Queue Status.
    Uses celery inspect to get active/reserved/scheduled tasks.
    """
    # Import celery_app lazily to avoid circular imports? 
    # Or assume it's available. 
    # We need to import it from where it's defined.
    try:
        from celery_app import celery_app
        i = celery_app.control.inspect()
        
        # Get stats
        active = i.active() or {}
        reserved = i.reserved() or {}
        scheduled = i.scheduled() or {}
        
        # Aggregate counts
        stats = {
            "active": sum(len(tasks) for tasks in active.values()),
            "reserved": sum(len(tasks) for tasks in reserved.values()),
            "scheduled": sum(len(tasks) for tasks in scheduled.values()),
            "workers": list(active.keys())
        }
    except Exception as e:
        stats = {"error": str(e)}
        active = {}
        
    return render_template('admin/queue.html', stats=stats, active_tasks=active)
