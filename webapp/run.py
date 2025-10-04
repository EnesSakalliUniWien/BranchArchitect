# --------------------------------------------------------------
#  run.py
# --------------------------------------------------------------
"""Development server runner for the webapp."""

import argparse
from typing import Any, Mapping, cast

try:
    # When running as a module (poetry run webapp)
    from webapp import create_app
except ImportError:
    # When running directly from the webapp directory
    from __init__ import create_app


def main():
    """Main entry point for the development server."""
    import sys
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    try:
        # Create app first so we can use its logger
        app = create_app()
        app.logger.info("[STARTUP] Flask app created successfully")

        config: Mapping[str, Any] = cast(Mapping[str, Any], app.config)
        debug_mode = bool(config.get("DEBUG", False))

        app.logger.info(
            f"[STARTUP] Starting server on {args.host}:{args.port} (debug={debug_mode})"
        )
        # *Never* enable `debug=True` for production – use a real WSGI/ASGI server
        app.run(host=args.host, port=args.port, debug=debug_mode)
    except Exception as e:
        # If app creation failed, fallback to stderr
        if "app" in locals() and hasattr(app, "logger"):
            app.logger.error(f"[ERROR] Failed to start server: {e}", exc_info=True)
        else:
            print(f"[ERROR] Failed to start server: {e}", file=sys.stderr)
            print(f"[ERROR] Traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
