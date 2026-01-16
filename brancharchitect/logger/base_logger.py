"""Base logging functionality for visualization and debugging."""

import logging
import base64
from typing import Any, cast, Callable, TypeVar
from functools import wraps

from brancharchitect.logger.html_content import CSS_LOG, MATH_JAX_HEADER

F = TypeVar("F", bound=Callable[..., Any])


class AlgorithmLogger:
    """Base logger class for algorithm visualization and debugging."""

    def __init__(self, name: str):
        self.name = name
        self.disabled = False
        self._html_content = ['<div class="content">']
        self._css_content: list[str] = []
        self._section_open = False

        # Include MathJax for mathematical notation
        self._include_mathjax = True
        self._mathjax_header = MATH_JAX_HEADER

        # Create logger with single handler to avoid duplication
        self.logger = logging.getLogger(name)

        # Only add a default StreamHandler if no handlers exist.
        # This prevents duplicate logs and avoids wiping handlers from other logger instances
        # that share the same name (e.g., when TableLogger is called inside MatrixLogger).
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            # Prevent propagation to avoid duplicate logs when a root logger is configured
            self.logger.propagate = False
            # Default to enabled, but specific subclasses or instances may override.
            self.disabled = False
        else:
            # If handlers exist, we still want to ensure a base level is set if it was NOTSET
            if self.logger.level == logging.NOTSET:
                self.logger.setLevel(logging.INFO)
            # We also ensure propagation is False to maintain AlgorithmLogger's behavior
            # of being a self-contained logger, unless the caller explicitly wants it.
            self.logger.propagate = False
            # If the logger was already configured (e.g. by another AlgorithmLogger),
            # we respect its current 'disabled' state if it has one, otherwise default to False.
            if not hasattr(self, "disabled"):
                self.disabled = False

        # Add enhanced CSS for basic display
        self._css_content.append(CSS_LOG)

    def section(self, title: str):
        """Create a new section in the log."""
        if self.disabled:
            return
        # Close any previously open section to keep HTML balanced
        if self._section_open:
            self._html_content.append("</section>")
            self._section_open = False

        self.logger.info(f"\n{'=' * 20} {title} {'=' * 20}\n")
        self._html_content.append(f'<section class="section"><h3>{title}</h3>')
        self._section_open = True

    def subsection(self, title: str):
        """Create a new subsection in the log."""
        if self.disabled:
            return
        self.logger.info(f"\n{'-' * 15} {title} {'-' * 15}\n")
        # Emit a standalone subsection header block to avoid unclosed tags
        self._html_content.append(f'<div class="subsection"><h4>{title}</h4></div>')

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

    def add_png_from_svg(self, svg_content: str, scale: float = 1.0) -> bool:
        """Convert SVG content to PNG and embed it as an image in the HTML log.

        Returns True on success, False if conversion fails (caller may fallback).
        """
        if self.disabled:
            return False

        try:
            # Import locally to avoid mandatory runtime dependency if not used
            import cairosvg  # type: ignore

            png_bytes = cairosvg.svg2png(
                bytestring=svg_content.encode("utf-8"),
                scale=scale,
                background_color="white",
            )
            b64 = base64.b64encode(png_bytes).decode("ascii")
            img_html = (
                '<div class="svg-container">'
                f'<img src="data:image/png;base64,{b64}" alt="Rendered tree visualization" '
                'style="max-width:100%; height:auto; display:block; margin:0 auto;"/>'
                "</div>"
            )
            self._html_content.append(img_html)
            return True
        except Exception:
            # On any error, return False so caller can fallback to SVG
            return False

    def end_section(self):
        """End the current section."""
        if self.disabled:
            return
        if self._section_open:
            self._html_content.append("</section>")
            self._section_open = False

    def clear(self):
        """Clear all accumulated content."""
        # Reset content to a clean initial state and restore default CSS
        self._html_content = ['<div class="content">']
        self._css_content = [CSS_LOG]
        self._section_open = False

    def get_html_content(self) -> str:
        """Get the accumulated HTML content with MathJax if enabled."""
        # Build a snapshot without mutating internal buffers
        parts = list(self._html_content)
        # Close any open section for the snapshot
        if self._section_open:
            parts.append("</section>")
        # Close the content wrapper
        parts.append("</div>")

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
            return self._mathjax_header + "\n" + "\n".join(parts) + toggle_script
        else:
            return "\n".join(parts) + toggle_script

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
