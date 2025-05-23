import os
import base64
import xml.etree.ElementTree as ET
from typing import Optional

# For SVG to PNG/PDF conversion
try:
    import cairosvg

    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False


def _svg_to_string(svg_root: ET.Element) -> str:
    try:
        return ET.tostring(svg_root, encoding="unicode")
    except Exception as e:
        return f"<svg><text fill='red'>Error creating SVG: {e}</text></svg>"


def _svg_to_png_base64(svg_string: str, scale: float = 1.0) -> Optional[str]:
    if not CAIRO_AVAILABLE:
        print("CairoSVG not installed. Please install with: pip install cairosvg")
        return None
    try:
        png_bytes = cairosvg.svg2png(bytestring=svg_string.encode("utf-8"), scale=scale)
        base64_data = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{base64_data}"
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return None


def _svg_to_pdf(svg_string: str, output_path: str, scale: float = 1.0) -> bool:
    if not CAIRO_AVAILABLE:
        print("CairoSVG not installed. Please install with: pip install cairosvg")
        return False
    try:
        # Patch: Ensure SVG has width/height attributes for CairoSVG
        if ('width="' not in svg_string) or ('height="' not in svg_string):
            try:
                root = ET.fromstring(svg_string)
                if "width" not in root.attrib:
                    root.set("width", "800")
                if "height" not in root.attrib:
                    root.set("height", "400")
                svg_string = ET.tostring(root, encoding="unicode")
            except Exception as e:
                print(f"Warning: Could not patch SVG size: {e}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cairosvg.svg2pdf(
            bytestring=svg_string.encode("utf-8"), write_to=output_path, scale=scale
        )
        return True
    except Exception as e:
        print(f"Error saving PDF: {e}")
        return False


# --- Public API ---
def save_to_pdf(svg_string: str, output_pdf: str) -> None:
    """
    Save the SVG as a PDF file. (Legacy interface, always returns None)
    """
    _svg_to_pdf(svg_string, output_pdf)


def save_pdf_dual_location(
    svg_string: str, user_pdf_path: str, project_export_name: Optional[str] = None
) -> str:
    save_to_pdf(svg_string, user_pdf_path)
    project_exports_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(project_exports_dir, exist_ok=True)
    if project_export_name is None:
        project_export_name = os.path.basename(user_pdf_path)
    project_pdf_path = os.path.join(project_exports_dir, project_export_name)
    save_to_pdf(svg_string, project_pdf_path)
    rel_path = os.path.relpath(project_pdf_path, start=os.getcwd())
    return rel_path


def create_copyable_svg_display(
    svg_string: str, pdf_path: Optional[str] = None
) -> None:
    """
    Display SVG content in a Jupyter notebook cell with copy, PNG, and PDF-as-PNG display.
    Ensures only a single output cell is produced in the notebook.
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML, Image
    import os

    items = []
    # SVG Preview and Copy
    output = widgets.Output()
    copy_button = widgets.Button(description="Copy SVG", button_style="info")
    text_area = widgets.Textarea(
        value=svg_string, layout=widgets.Layout(width="100%", height="120px")
    )
    text_area.layout.display = "none"

    def on_copy_clicked(b):
        try:
            import pyperclip
            pyperclip.copy(svg_string)
            copy_button.description = "Copied!"
        except Exception:
            text_area.layout.display = "block"
            copy_button.description = "Manual Copy Below"

    copy_button.on_click(on_copy_clicked)
    with output:
        display(HTML(svg_string))
    svg_box = widgets.VBox([output, widgets.HBox([copy_button]), text_area])
    items.append(("SVG Preview", svg_box))

    # PNG Preview and Copy
    png_b64 = _svg_to_png_base64(svg_string)
    if png_b64:
        png_output = widgets.Output()
        with png_output:
            img_html = f'<img src="{png_b64}" style="max-width:100%;"/>'
            display(HTML(img_html))
        copy_png_button = widgets.Button(
            description="Copy PNG (base64)", button_style="info"
        )
        png_status = widgets.Label("")
        def on_copy_png(b):
            try:
                import pyperclip
                pyperclip.copy(png_b64)
                copy_png_button.description = "Copied!"
            except Exception:
                png_status.value = "Manual copy below"
        copy_png_button.on_click(on_copy_png)
        png_box = widgets.VBox([png_output, widgets.HBox([copy_png_button, png_status])])
        items.append(("PNG Preview", png_box))

    # PDF Preview (as PNG)
    if pdf_path and os.path.isfile(pdf_path):
        pdf_output = widgets.Output()
        with pdf_output:
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(pdf_path, dpi=300)
                if pages:
                    tmp_png = "/tmp/_copyable_pdf_page_1.png"
                    pages[0].save(tmp_png, "PNG")
                    display(Image(filename=tmp_png))
            except Exception as e:
                display(HTML(f'<div style="color:red">PDF to PNG preview failed: {e}</div>'))
        items.append(("PDF Preview", pdf_output))

    # Accordion for all outputs
    accordion = widgets.Accordion(children=[item[1] for item in items])
    for idx, item in enumerate(items):
        accordion.set_title(idx, item[0])
    display(accordion)


def create_pdf_display(
    pdf_path: str, width: int = 800, height: int = 400, show_png: bool = False
) -> None:
    import ipywidgets as widgets
    from IPython.display import display, IFrame, HTML
    import shutil

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    notebooks_dir = os.path.join(project_root, "notebooks")
    exports_dir = os.path.join(notebooks_dir, "exports")
    os.makedirs(exports_dir, exist_ok=True)
    export_pdf_path = os.path.join(exports_dir, os.path.basename(pdf_path))
    shutil.copy2(pdf_path, export_pdf_path)
    rel_path = f"exports/{os.path.basename(pdf_path)}"
    try:
        display(IFrame(src=rel_path, width=width, height=height))
    except Exception as e:
        print(
            f"IFrame failed: {e}, falling back to IPython.display (PDF not available)"
        )
    if os.path.exists(export_pdf_path):
        with open(export_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        download_button = widgets.Button(
            description="Download PDF", button_style="success"
        )

        def on_download_clicked(b):
            b64 = base64.b64encode(pdf_bytes).decode()
            payload = f"data:application/pdf;base64,{b64}"
            display(
                HTML(
                    f'<a download="{os.path.basename(export_pdf_path)}" href="{payload}" target="_blank">Click here to download PDF</a>'
                )
            )

        download_button.on_click(on_download_clicked)
        display(download_button)
    if show_png:
        try:
            png_bytes = cairosvg.pdf2png(url=export_pdf_path)
            png_b64 = base64.b64encode(png_bytes).decode()
            img_html = f'<img src="data:image/png;base64,{png_b64}" width="{width}"/>'
            display(HTML(img_html))
            copy_button = widgets.Button(description="Copy PNG", button_style="info")

            def on_copy_png(b):
                display(HTML(f'<textarea style="width:100%">{png_b64}</textarea>'))
                copy_button.description = "Manual Copy Below"

            copy_button.on_click(on_copy_png)
            display(copy_button)
        except Exception as e:
            display(HTML(f'<div style="color:red">PNG preview failed: {e}</div>'))


def create_svg_root(
    width: float, height: float, background_color: str = "#FFFFFF"
) -> ET.Element:
    """Creates the root SVG element."""
    svg_root = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1",
        width=str(width),
        height=str(height),
        viewBox=f"0 0 {width} {height}",
    )
    # Add a background rectangle if a color is specified
    if background_color:
        ET.SubElement(
            svg_root,
            "rect",
            width="100%",
            height="100%",
            fill=background_color,
        )
    return svg_root


def create_container_group(
    svg_root: ET.Element, margin_left: float, margin_top: float
) -> ET.Element:
    """Creates the main container group with margins."""
    container = ET.SubElement(
        svg_root, "g", transform=f"translate({margin_left}, {margin_top})"
    )
    return container