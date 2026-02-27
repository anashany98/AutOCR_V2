from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from modules.auth_manager import AuthManager
from web_app.services import get_db

auth_bp = Blueprint("auth", __name__)


def get_auth_manager():
    return AuthManager(get_db())


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        auth = get_auth_manager()
        user = auth.verify_login(username, password)

        if user:
            login_user(user)
            return redirect(url_for("main.index"))

        if getattr(auth, "last_error", None) == "email_not_verified":
            flash("Debes verificar tu email antes de iniciar sesion.", "warning")
        else:
            flash("Usuario o contrasena incorrectos", "danger")

    return render_template("login.html")


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        role_selection = request.form.get("role_type")

        auth = get_auth_manager()
        success, res = auth.create_user_with_email(
            username,
            email,
            password,
            role=role_selection,
            allow_elevated_role=False,
        )

        if success:
            # res is token
            from modules.email_sender import EmailSender
            from web_app.services import load_configuration

            config = load_configuration()
            sender = EmailSender(config)

            # Use request.url_root for base url
            base_url = request.url_root.rstrip("/")
            sent = sender.send_verification_email(email, res, base_url)

            if sent:
                flash("Cuenta creada. Revisa tu email para verificarla.", "info")
            else:
                flash("Cuenta creada, pero fallo el envio del email. Contacta soporte.", "warning")

            return redirect(url_for("auth.login"))

        flash(f"Error: {res}", "danger")

    return render_template("register.html")


@auth_bp.route("/verify/<token>")
def verify_email(token):
    auth = get_auth_manager()
    success, msg = auth.verify_email(token)
    if success:
        flash(msg, "success")
        return redirect(url_for("auth.login"))

    flash(msg, "danger")
    return redirect(url_for("auth.login"))


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        auth = get_auth_manager()
        success, _msg, token = auth.request_password_reset(email)

        if success and token:
            from modules.email_sender import EmailSender
            from web_app.services import load_configuration

            sender = EmailSender(load_configuration())
            base_url = request.url_root.rstrip("/")
            sender.send_reset_email(email, token, base_url)

        # Avoid account enumeration.
        flash("Si el email existe, recibiras instrucciones.", "info")
        return redirect(url_for("auth.login"))

    return render_template("forgot_password.html")


@auth_bp.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if request.method == "POST":
        password = request.form.get("password")
        auth = get_auth_manager()
        success, msg = auth.reset_password(token, password)
        if success:
            flash(msg, "success")
            return redirect(url_for("auth.login"))

        flash(msg, "danger")

    return render_template("reset_password.html", token=token)
