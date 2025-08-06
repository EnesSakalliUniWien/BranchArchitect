"""Base logging functionality for visualization and debugging."""

import logging
import re
from typing import Any, cast, Callable, TypeVar
from functools import wraps

from brancharchitect.core.html_content import (
    CSS_LOG,
    MATH_JAX_HEADER,
)

F = TypeVar("F", bound=Callable[..., Any])


def format_set(s: Any) -> str:
    """Format set for consistent display."""
    if not s:
        return "∅"
    return "{" + ", ".join(str(x) for x in sorted(s)) + "}"


def beautify_frozenset(obj: Any) -> str:
    """Convert frozenset objects to beautiful mathematical notation."""
    if obj is None:
        return "∅"

    # Convert to string first
    s = str(obj)

    # Replace frozenset({...}) with just {...}
    s = re.sub(r"frozenset\(\{(.+?)\}\)", r"{\1}", s)

    # Replace nested frozensets recursively
    while "frozenset" in s:
        s = re.sub(r"frozenset\(\{(.+?)\}\)", r"{\1}", s)

    # Replace double parentheses around single elements
    s = re.sub(r"\(\((.+?)\)\)", r"(\1)", s)

    # Clean up any remaining artifacts
    s = s.replace("()", "∅")
    s = s.replace("{}", "∅")

    return s


class AlgorithmLogger:
    """Base logger class for algorithm visualization and debugging."""

    def __init__(self, name: str):
        self.name = name
        self.disabled = False
        self._html_content = ['<div class="content">']
        self._css_content: list[str] = []

        # Include MathJax for mathematical notation
        self._include_mathjax = True
        self._mathjax_header = MATH_JAX_HEADER

        # Create logger with single handler to avoid duplication
        self.logger = logging.getLogger(name)
        self.logger.handlers = []  # Clear any existing handlers

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

        # Add enhanced CSS for basic display
        self._css_content.append(CSS_LOG)

    def section(self, title: str):
        """Create a new section in the log."""
        if self.disabled:
            return
        self.logger.info(f"\n{'=' * 20} {title} {'=' * 20}\n")
        self._html_content.append(f'<section class="section"><h3>{title}</h3>')

    def subsection(self, title: str):
        """Create a new subsection in the log."""
        if self.disabled:
            return
        self.logger.info(f"\n{'-' * 15} {title} {'-' * 15}\n")
        self._html_content.append(f'<div class="subsection"><h4>{title}</h4>')

    def info(self, message: str):
        """Log info message."""
        if self.disabled:
            return
        self.logger.info(message)
        self._html_content.append(f'<p class="info">{message}</p>')

    def warning(self, message: str):
        """Log warning message."""
        if self.disabled:
            return
        self.logger.warning(message)
        self._html_content.append(f'<p class="warning">{message}</p>')

    def error(self, message: str):
        """Log an error message."""
        if self.disabled:
            return
        self.logger.error(message)
        self._html_content.append(f'<p class="error">{message}</p>')

    def debug(self, message: str):
        """Log debug message."""
        if self.disabled:
            return
        self.logger.debug(message)
        self._html_content.append(f'<p class="debug">{message}</p>')

    def result(self, label: str, value: Any):
        """Log a result with a label."""
        if self.disabled:
            return
        self.logger.info(f"{label}: {value}")
        self._html_content.append(
            f'<div class="result"><strong>{label}:</strong> {value}</div>'
        )

    def raw_html(self, html_content: str):
        """Add raw HTML content to the debug output."""
        if self.disabled:
            return
        self._html_content.append(html_content)

    def html(self, html_content: str):
        """Add HTML content to the debug output (alias for raw_html)."""
        self.raw_html(html_content)

    def add_css(self, css: str):
        """Add custom CSS to the debug output."""
        if self.disabled:
            return
        self._css_content.append(css)

    def add_svg(self, svg_content: str):
        """Add SVG visualization to the debug output."""
        if self.disabled:
            return
        self._html_content.append(f'<div class="svg-container">{svg_content}</div>')

    def end_section(self):
        """End the current section."""
        if self.disabled:
            return
        self._html_content.append("</section>")

    def clear(self):
        """Clear all accumulated content."""
        self._html_content = []
        self._css_content = []

    def get_html_content(self) -> str:
        """Get the accumulated HTML content with MathJax if enabled."""
        # Close all open sections and the content div
        self._html_content.append("</section>")
        self._html_content.append("</div>")

        # Add toggle script at the end
        toggle_script = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Set up matrix toggle buttons
            const toggleButtons = document.querySelectorAll('.matrix-toggle .toggle-button');
            toggleButtons.forEach(button => {
                button.addEventListener('click', function() {
                    // Find the parent matrix container
                    const matrixContainer = this.closest('.matrix-container');
                    if (!matrixContainer) return;

                    // Get the view to show
                    const viewType = this.getAttribute('data-view');

                    // Update active button state
                    const buttons = matrixContainer.querySelectorAll('.toggle-button');
                    buttons.forEach(btn => btn.classList.remove('active'));
                    this.classList.add('active');

                    // Hide all views in this container
                    const views = matrixContainer.querySelectorAll('.matrix-view');
                    views.forEach(view => view.style.display = 'none');

                    // Show the selected view
                    const selectedView = matrixContainer.querySelector('.' + viewType + '-view');
                    if (selectedView) {
                        selectedView.style.display = 'block';

                        // If it's MathJax, trigger typesetting
                        if (viewType === 'mathjax' && typeof MathJax !== 'undefined') {
                            MathJax.typeset([selectedView]);
                        }
                    }
                });
            });
        });
        </script>
        """

        # If MathJax is enabled, include the header at the beginning
        if self._include_mathjax:
            return (
                self._mathjax_header
                + "\n"
                + "\n".join(self._html_content)
                + toggle_script
            )
        else:
            return "\n".join(self._html_content) + toggle_script

    def get_css_content(self) -> str:
        """Get the accumulated CSS content."""
        return "\n".join(self._css_content)

    def log_execution(self, func: F) -> F:
        """Decorator for logging function execution with type safety."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.section(f"Executing {func.__name__}")
            try:
                result = func(*args, **kwargs)
                self.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                self.info(f"Error in {func.__name__}: {str(e)}")
                raise

        return cast(F, wrapper)
