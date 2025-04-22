import os
import cairosvg
from IPython.display import IFrame
from typing import Optional

def save_to_pdf(
    svg_string: str, output_pdf: str, enable_latex: bool, use_mathjax: bool
) -> None:
    """
    Attempt to save the SVG as a PDF file.xw

    Args:
        svg_string: SVG content as string
        output_pdf: Path to save the PDF file
        enable_latex: Whether LaTeX rendering is enabled
        use_mathjax: Whether to use MathJax for browser-based rendering
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        # Convert SVG to PDF and save
        cairosvg.svg2pdf(bytestring=svg_string.encode("utf-8"), write_to=output_pdf)
        if enable_latex and use_mathjax:
            print(
                f"Warning: PDF saved to {output_pdf}, but LaTeX rendering via MathJax is only visible in SVG viewers/browsers, not in this directly converted PDF. Raw LaTeX commands will appear."
            )
        else:
            print(f"SVG successfully converted and saved to {output_pdf}")
    except ImportError:
        print(
            "Warning: cairosvg not found. Please install it (`pip install cairosvg`) to save as PDF."
        )
    except Exception as e:
        print(f"Warning: Failed to convert SVG to PDF: {e}")


def create_copyable_svg_display(svg_string: str) -> None:
    """Display SVG content in a Jupyter notebook cell."""
    from IPython.display import display, HTML

    display(HTML(svg_string))


def create_pdf_display(pdf_path: str, width: int = 800, height: int = 400) -> None:
    """
    Display a PDF file in a Jupyter notebook cell using an IFrame.
    
    Args:
        pdf_path: Path to the PDF file to display
        width: Width of the displayed PDF in pixels (default: 800)
        height: Height of the displayed PDF in pixels (default: 400)
    """
    return IFrame(src=pdf_path, width=width, height=height)

