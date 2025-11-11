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
    svg,
    pdf_path: Optional[str] = None,
    *,
    show_png: bool = False,
    show_pdf_preview: bool = False,
    lazy: bool = True,
    return_widget: bool = False,
):
    """
    Display SVG content in a Jupyter notebook with lightweight panels.

    Goals:
    - Avoid duplicating large SVG strings in widget state or outputs
    - Make PNG/PDF previews optional and lazily generated
    - Allow callers to obtain the widget and manage display themselves

    Args:
        svg: SVG as a string or xml.etree.ElementTree.Element
        pdf_path: Optional path to a generated PDF for preview
        show_png: Whether to include a PNG preview panel (default False)
        show_pdf_preview: Whether to include a PDF-as-PNG preview panel (default False)
        lazy: Generate heavy previews only on button click (default True)
        return_widget: If True, return the Accordion instead of displaying it
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML, Image
    import os
    import xml.etree.ElementTree as ET

    # Normalize input to string
    if isinstance(svg, ET.Element):
        svg_string = _svg_to_string(svg)
    else:
        svg_string = str(svg)

    items = []

    # SVG Preview with copy/download controls
    svg_output = widgets.Output()
    with svg_output:
        display(HTML(svg_string))

    copy_button = widgets.Button(description="Copy SVG", button_style="info")
    copy_status = widgets.Label("")
    download_button = widgets.Button(description="Download SVG", button_style="success")

    def on_copy_clicked(_):
        try:
            import pyperclip  # type: ignore
            pyperclip.copy(svg_string)
            copy_button.description = "Copied!"
            copy_status.value = ""
        except Exception:
            # Avoid embedding a huge textarea by default; guide the user instead
            copy_status.value = "Install pyperclip to enable clipboard copy."

    copy_button.on_click(on_copy_clicked)

    def on_download_clicked(_):
        # Provide a one-time HTML anchor using a data URL. Only generated if the user asks.
        b64 = base64.b64encode(svg_string.encode("utf-8")).decode("ascii")
        href = f"data:image/svg+xml;base64,{b64}"
        link = HTML(
            f'<a download="figure.svg" href="{href}" target="_blank">Click to download SVG</a>'
        )
        display(link)

    download_button.on_click(on_download_clicked)

    svg_controls = widgets.HBox([copy_button, download_button, copy_status])
    items.append(("SVG Preview", widgets.VBox([svg_output, svg_controls])))

    # PNG Preview (optional)
    if show_png:
        png_panel_output = widgets.Output()
        if lazy:
            gen_png_btn = widgets.Button(description="Generate PNG preview", button_style="info")

            def on_gen_png(_):
                png_b64 = _svg_to_png_base64(svg_string)
                if png_b64:
                    with png_panel_output:
                        display(HTML(f'<img src="{png_b64}" style="max-width:100%;"/>'))

            gen_png_btn.on_click(on_gen_png)
            items.append(("PNG Preview", widgets.VBox([gen_png_btn, png_panel_output])))
        else:
            png_b64 = _svg_to_png_base64(svg_string)
            if png_b64:
                with png_panel_output:
                    display(HTML(f'<img src="{png_b64}" style="max-width:100%;"/>'))
                items.append(("PNG Preview", png_panel_output))

    # PDF Preview (as PNG, optional)
    if show_pdf_preview and pdf_path and os.path.isfile(pdf_path):
        pdf_panel_output = widgets.Output()
        if lazy:
            gen_pdf_btn = widgets.Button(description="Generate PDF preview", button_style="info")

            def on_gen_pdf(_):
                try:
                    from pdf2image import convert_from_path  # type: ignore
                    pages = convert_from_path(pdf_path, dpi=300)
                    if pages:
                        tmp_png = "/tmp/_copyable_pdf_page_1.png"
                        pages[0].save(tmp_png, "PNG")
                        with pdf_panel_output:
                            display(Image(filename=tmp_png))
                except Exception as e:
                    with pdf_panel_output:
                        display(HTML(f'<div style="color:red">PDF preview failed: {e}</div>'))

            gen_pdf_btn.on_click(on_gen_pdf)
            items.append(("PDF Preview", widgets.VBox([gen_pdf_btn, pdf_panel_output])))
        else:
            try:
                from pdf2image import convert_from_path  # type: ignore
                pages = convert_from_path(pdf_path, dpi=300)
                if pages:
                    tmp_png = "/tmp/_copyable_pdf_page_1.png"
                    pages[0].save(tmp_png, "PNG")
                    with pdf_panel_output:
                        display(Image(filename=tmp_png))
                items.append(("PDF Preview", pdf_panel_output))
            except Exception as e:
                items.append(("PDF Preview", HTML(f'<div style="color:red">PDF preview failed: {e}</div>')))

    # Build accordion
    accordion_children = [panel for _, panel in items]
    accordion = widgets.Accordion(children=accordion_children)
    for idx, (title, _) in enumerate(items):
        accordion.set_title(idx, title)

    if return_widget:
        return accordion
    else:
        display(accordion)
        return None


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
