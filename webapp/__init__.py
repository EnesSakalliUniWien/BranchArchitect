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

    app: Flask | None = None
    try:
        # Create app first to have access to logger
        app = Flask(__name__, static_folder="static")
        app.config.from_object(Config)

        # Configure logging early to capture all messages
        configure_logging(app)

        app.logger.info("[INIT] Creating Flask instance...")
        app.logger.info("[INIT] Loading config...")

        app.logger.info("[INIT] Enabling CORS...")
        # Enable CORS
        CORS(app)

        app.logger.info("[INIT] Registering blueprints...")
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

        app.logger.info("[INIT] Flask app creation complete")
        return app
    except Exception as e:
        # If logging is not configured yet, fallback to stderr
        if app is not None and hasattr(app, "logger"):
            app.logger.error(f"[INIT ERROR] Failed to create app: {e}", exc_info=True)
        else:
            print(f"[INIT ERROR] Failed to create app: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
        raise
