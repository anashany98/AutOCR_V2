from flask import Blueprint, render_template, jsonify, request, current_app
from web_app.services import get_logger
from flask_wtf.csrf import CSRFError
from werkzeug.exceptions import HTTPException

errors_bp = Blueprint('errors', __name__)

@errors_bp.app_errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Resource not found", "code": 404}), 404
    return render_template('errors/404.html'), 404

@errors_bp.app_errorhandler(500)
def internal_error(error):
    get_logger().error(f"Internal Server Error: {error}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal server error", "code": 500}), 500
    return render_template('errors/500.html'), 500

@errors_bp.app_errorhandler(CSRFError)
def handle_csrf_error(error):
    # Return JSON for API consumers; HTML for browser navigation.
    if request.path.startswith('/api/') or request.accept_mimetypes.best == 'application/json':
        return jsonify({"error": "CSRF token missing or invalid", "code": 403}), 403
    return render_template('errors/generic_error.html', error=error), 403

@errors_bp.app_errorhandler(Exception)
def handle_unexpected_exception(error):
    if isinstance(error, HTTPException):
        if request.path.startswith('/api/'):
            return jsonify({"error": error.description, "code": error.code}), error.code
        return error
    get_logger().error(f"Unhandled Exception: {error}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(error) if current_app.debug else None
        }), 500
    return render_template('errors/generic_error.html', error=error), 500
