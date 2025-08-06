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
        print(f"[STARTUP] Creating Flask app...", file=sys.stderr)
        app = create_app()
        print(f"[STARTUP] Flask app created successfully", file=sys.stderr)
        
        config: Mapping[str, Any] = cast(Mapping[str, Any], app.config)
        debug_mode = bool(config.get("DEBUG", False))
        
        print(f"[STARTUP] Starting server on {args.host}:{args.port} (debug={debug_mode})", file=sys.stderr)
        # *Never* enable `debug=True` for production – use a real WSGI/ASGI server
        app.run(host=args.host, port=args.port, debug=debug_mode)
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}", file=sys.stderr)
        print(f"[ERROR] Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
