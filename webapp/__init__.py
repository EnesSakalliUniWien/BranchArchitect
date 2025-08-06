# --------------------------------------------------------------
#  __init__.py (package root)
# --------------------------------------------------------------
from flask import Flask
from flask_cors import CORS

from .config import Config
from .services.logging_config import configure_logging
from .routes.routes import bp as main_bp
from pathlib import Path

# Inject config into Jinja globals so templates can access constants
from typing import cast

__all__ = ["create_app"]


def create_app() -> Flask:
    """Factory for the Flask WSGI application.

    Using a *factory* makes unit testing trivial (each test just calls
    ``create_app()``) and prevents module-level side effects.
    """
    import sys
    
    try:
        print("[INIT] Creating Flask instance...", file=sys.stderr)
        app = Flask(__name__, static_folder="static")
        
        print("[INIT] Loading config...", file=sys.stderr)
        app.config.from_object(Config)
        
        print("[INIT] Enabling CORS...", file=sys.stderr)
        # Enable CORS
        CORS(app)

        print("[INIT] Configuring logging...", file=sys.stderr)
        # Logging first so early import failures are visible
        configure_logging(app)

        print("[INIT] Registering blueprints...", file=sys.stderr)
        # Register blueprints (keeps route definitions in *one* place)
        app.register_blueprint(main_bp)

        app.jinja_env.globals = cast(dict[str, object], app.jinja_env.globals)  # type: ignore
        app.jinja_env.globals.update(config=app.config)

        # Expose short *commit* string if present (useful in sentry etc.)
        try:
            with open(Path(__file__).resolve().parent.parent / "commithash", "r") as fp:
                app.config["APP_COMMIT"] = fp.read().strip()
        except FileNotFoundError:
            app.config["APP_COMMIT"] = "unknown"
        
        print("[INIT] Flask app creation complete", file=sys.stderr)
        return app
    except Exception as e:
        print(f"[INIT ERROR] Failed to create app: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise