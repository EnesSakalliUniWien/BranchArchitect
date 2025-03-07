from . import jt_logger, log_tree_splits
import traceback
import sys
from typing import Dict, Any, Optional
from functools import wraps
from datetime import datetime


def log_stacktrace(exception):
    """Log a detailed stack trace for an exception."""
    try:
        # Get the full stack trace
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
        formatted_trace = "".join(stack_trace)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log to HTML output with sections and formatting
        jt_logger.section("Stack Trace ====================")
        jt_logger.raw_html(
            f"""
            <div class="error-block">
                <h4>Error at {timestamp}</h4>
                <div class="error-message">{str(exception)}</div>
                <pre class="stack-trace">{formatted_trace}</pre>
            </div>
            """
        )
        # Also log to console if enabled
        print(f"Error: {str(exception)}", file=sys.stderr)
        print(formatted_trace, file=sys.stderr)

    except Exception as e:
        # Fallback error handling
        print(f"Error in error handling: {str(e)}", file=sys.stderr)


def log_detailed_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log a detailed error with context information."""
    try:
        jt_logger.section("ERROR DETAILS")

        # Log the error message and type
        error_type = type(error).__name__
        error_msg = str(error)

        jt_logger.raw_html(
            f"""
            <div class="error-container">
                <h3 style="color: #d9534f; margin-top: 0;">Error Detected: {error_type}</h3>
                <p><strong>Message:</strong> {error_msg}</p>
            </div>
            """
        )

        # Log tree splits if available in context
        if context and "tree1" in context and "tree2" in context:
            jt_logger.section("Tree Split Analysis at Error")
            log_tree_splits(context["tree1"], context["tree2"])

        # Log the stack trace
        tb_str = "".join(traceback.format_tb(error.__traceback__))
        jt_logger.raw_html(f"""<pre class="stack-trace"><code>{tb_str}</code></pre>""")

        # Log additional context
        if context:
            jt_logger.section("Error Context")
            for key, value in context.items():
                if key not in ("tree1", "tree2"):  # Skip trees as they're handled above
                    jt_logger.info(f"{key}: {value}")

    except Exception as e:
        # Fallback error handling
        print(f"Error in error logging: {str(e)}", file=sys.stderr)
        print(f"Original error: {str(error)}", file=sys.stderr)


def debug_algorithm_execution(func):
    """Decorator to wrap algorithm execution with debugging."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            jt_logger.section(f"Running {func.__name__}")
            jt_logger.info(f"Function: {func.__name__}")

            # Execute the function
            result = func(*args, **kwargs)

            jt_logger.info(f"Completed {func.__name__} successfully")
            return result

        except Exception as e:
            # Log the error with detailed information
            log_detailed_error(
                e,
                {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs": str(kwargs.keys()),
                },
            )
            raise

    return wrapper
