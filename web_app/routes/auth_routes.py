from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from web_app.services import get_db
from modules.auth_manager import AuthManager

auth_bp = Blueprint('auth', __name__)

def get_auth_manager():
    return AuthManager(get_db())

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        auth = get_auth_manager()
        user = auth.verify_login(username, password)
        
        if user:
            login_user(user)
            return redirect(url_for('main.index'))
        else:
            flash('Usuario o contraseña incorrectos', 'danger')
            
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    # Only admins or special setup can register for now
    # Or keep it open for demo
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        auth = get_auth_manager()
        success, msg = auth.create_user(username, password)
        
        if success:
            flash('Usuario creado. Por favor inicia sesión.', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash(f'Error: {msg}', 'danger')
            
    return render_template('login.html', register_mode=True)
