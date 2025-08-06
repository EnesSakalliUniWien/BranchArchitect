"""
Unified display utilities for safely displaying plots in Jupyter notebooks.
Handles file existence checks and provides consistent error handling.
"""

import time
from typing import Optional, Dict, Any
from pathlib import Path
from IPython.display import Image, IFrame, display, HTML


def safe_display_image(
    filename: str, 
    width: Optional[int] = None, 
    height: Optional[int] = None,
    retry_count: int = 3,
    retry_delay: float = 0.5,
    fallback_message: bool = True
) -> None:
    """
    Safely display an image file with retry logic and error handling.
    
    Args:
        filename: Path to the image file (PNG, JPG, SVG, etc.)
        width: Display width in pixels
        height: Display height in pixels  
        retry_count: Number of times to retry if file not found
        retry_delay: Delay in seconds between retries
        fallback_message: Whether to show an error message on failure
    """
    path = Path(filename)
    
    # Retry logic for file generation
    for i in range(retry_count):
        if path.exists():
            try:
                img = Image(filename=str(path))
                if width:
                    img.width = width
                if height:
                    img.height = height
                display(img)
                return
            except Exception as e:
                if fallback_message:
                    display(HTML(f'<div style="color:red">Error displaying image: {e}</div>'))
                return
        
        if i < retry_count - 1:
            time.sleep(retry_delay)
    
    # File not found after retries
    if fallback_message:
        display(HTML(
            f'<div style="color:red; border: 1px solid red; padding: 10px;">'
            f'<b>Image not found:</b> {filename}<br>'
            f'<small>The file may still be generating or the path may be incorrect.</small>'
            f'</div>'
        ))


def safe_display_pdf(
    filename: str,
    width: int = 800,
    height: int = 600,
    display_method: str = "auto",
    retry_count: int = 3,
    retry_delay: float = 0.5,
    fallback_message: bool = True
) -> None:
    """
    Safely display a PDF file with multiple display methods.
    
    Args:
        filename: Path to the PDF file
        width: Display width in pixels
        height: Display height in pixels
        display_method: "iframe", "png", "auto" (tries iframe then png)
        retry_count: Number of times to retry if file not found
        retry_delay: Delay in seconds between retries
        fallback_message: Whether to show an error message on failure
    """
    path = Path(filename)
    
    # Retry logic for file generation
    for i in range(retry_count):
        if path.exists():
            break
        if i < retry_count - 1:
            time.sleep(retry_delay)
    
    if not path.exists():
        if fallback_message:
            display(HTML(
                f'<div style="color:red; border: 1px solid red; padding: 10px;">'
                f'<b>PDF not found:</b> {filename}<br>'
                f'<small>The file may still be generating or the path may be incorrect.</small>'
                f'</div>'
            ))
        return
    
    if display_method == "iframe" or display_method == "auto":
        try:
            display(IFrame(str(path), width=width, height=height))
            return
        except Exception as e:
            if display_method == "iframe":
                if fallback_message:
                    display(HTML(f'<div style="color:red">Error displaying PDF: {e}</div>'))
                return
    
    if display_method == "png" or display_method == "auto":
        try:
            # Try to convert PDF to PNG for display
            from pdf2image import convert_from_path
            
            images = convert_from_path(str(path), dpi=150)
            if images:
                # Display first page
                display(images[0])
                if len(images) > 1:
                    display(HTML(f'<small>Showing page 1 of {len(images)}</small>'))
            return
        except ImportError:
            if fallback_message:
                display(HTML(
                    '<div style="color:orange">pdf2image not installed. '
                    'Install with: pip install pdf2image</div>'
                ))
        except Exception as e:
            if fallback_message:
                display(HTML(f'<div style="color:red">Error converting PDF: {e}</div>'))


def display_plot_output(
    output_path: str,
    display_options: Optional[Dict[str, Any]] = None,
    retry_count: int = 3,
    retry_delay: float = 0.5
) -> None:
    """
    Unified function to display plot outputs regardless of format.
    
    Args:
        output_path: Path to the output file
        display_options: Dictionary with display options:
            - width: Display width
            - height: Display height
            - method: Display method for PDFs
        retry_count: Number of times to retry if file not found
        retry_delay: Delay in seconds between retries
    """
    if display_options is None:
        display_options = {}
    
    path = Path(output_path)
    ext = path.suffix.lower()
    
    if ext in ['.png', '.jpg', '.jpeg', '.svg']:
        safe_display_image(
            filename=str(path),
            width=display_options.get('width'),
            height=display_options.get('height'),
            retry_count=retry_count,
            retry_delay=retry_delay
        )
    elif ext == '.pdf':
        safe_display_pdf(
            filename=str(path),
            width=display_options.get('width', 800),
            height=display_options.get('height', 600),
            display_method=display_options.get('method', 'auto'),
            retry_count=retry_count,
            retry_delay=retry_delay
        )
    else:
        display(HTML(
            f'<div style="color:orange">Unsupported file type: {ext}</div>'
        ))


def check_output_file(output_path: str, timeout: float = 5.0) -> bool:
    """
    Check if output file exists, with timeout for file generation.
    
    Args:
        output_path: Path to check
        timeout: Maximum time to wait for file to appear
        
    Returns:
        True if file exists, False otherwise
    """
    path = Path(output_path)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if path.exists():
            # Give a bit more time for file to be fully written
            time.sleep(0.1)
            return True
        time.sleep(0.1)
    
    return False


def generate_output_path(
    base_name: str,
    format: str = "png",
    output_dir: Optional[str] = None,
    timestamp: bool = False
) -> str:
    """
    Generate a safe output path for plots.
    
    Args:
        base_name: Base name for the file
        format: File format (png, pdf, svg)
        output_dir: Output directory (creates if needed)
        timestamp: Whether to add timestamp to filename
        
    Returns:
        Full path to output file
    """
    if output_dir is None:
        output_dir = "."
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean base name
    base_name = "".join(c for c in base_name if c.isalnum() or c in "-_")
    
    if timestamp:
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{base_name}_{timestamp_str}"
    
    return str(output_path / f"{base_name}.{format}")