from web_app.app import app, init_app
import os

# Initialize application components
# In production, this runs once per worker
init_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
