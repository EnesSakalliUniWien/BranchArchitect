# Utility functions for LaTeX/MathJax handling, moved from paper_plots.py

def check_latex_rendering_mode(enable_latex, use_mathjax, output_pdf):
    """
    Check the rendering mode for LaTeX/MathJax based on the provided flags.
    """
    if enable_latex:
        if use_mathjax:
            return "mathjax"
        elif output_pdf:
            return "pdf"
        else:
            return "latex"
    return None

def add_mathjax_to_svg(svg_root):
    """
    Add MathJax configuration and source scripts to the SVG root element.
    """
    if not mathjax_already_in_svg(svg_root):
        add_mathjax_config_script(svg_root)
        add_mathjax_source_script(svg_root)

def mathjax_already_in_svg(svg_root):
    """
    Check if MathJax scripts are already present in the SVG root element.
    """
    for child in svg_root:
        if child.tag == "script" and "MathJax" in child.attrib.get("src", ""):
            return True
    return False

def add_mathjax_config_script(svg_root):
    """
    Add MathJax configuration script to the SVG root element.
    """
    config_script = svg_root.makeelement("script", {})
    config_script.text = """
    MathJax.Hub.Config({
        SVG: {
            scale: 100
        }
    });
    """
    svg_root.append(config_script)

def add_mathjax_source_script(svg_root):
    """
    Add MathJax source script to the SVG root element.
    """
    source_script = svg_root.makeelement("script", {"src": "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"})
    svg_root.append(source_script)