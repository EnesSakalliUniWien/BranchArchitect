# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Default Layout Constants
DEFAULT_H_SPACING_PER_LEAF = 15
DEFAULT_V_SPACING_PER_DEPTH = 50
DEFAULT_LABEL_AREA_HEIGHT = 50

CAPTION_HEIGHT = 15

INTER_TREE_SPACING = 1

MARGIN_X = 2

MARGIN_Y = 10

ZERO_LENGTH_TOLERANCE = 1e-9

FOOTER_FONT_SIZE_ADD = 1

# --- Default Layout Constants ---
ZERO_LENGTH_TOLERANCE = 1e-9  # Threshold below which branch lengths are treated as zero.
DEFAULT_CLADOGRAM_V_SPACING = 50 # Default vertical space between depth levels in cladograms.
DEFAULT_TARGET_HEIGHT_BASE = 150 # Minimum target height for phylograms.
DEFAULT_TARGET_HEIGHT_PER_DEPTH = 50 # Added height per depth level for default phylogram height.


# Font Constantsw
FONT_SANS_SERIF = "'Helvetica Neue', Helvetica, Arial, sans-serif"
FONT_SERIF = "Georgia, Times, 'Times New Roman', serif"
DEFAULT_NODE_LENGTH = 1.0

# Style Defaults
DEFAULT_SPLIT_DASH = True
DEFAULT_ENCLOSE_PADDING = 8
DEFAULT_ENCLOSE_RX = 5
DEFAULT_ENCLOSE_STROKE_WIDTH = "1.5"
DEFAULT_ENCLOSE_DASHARRAY = "none"

# Footer related constants
FOOTER_HEIGHT = 10  # Default height for the footer area
FOOTER_PADDING = 5  # Padding between tree and footer text
FOOTER_FONT_SIZE_ADD = 1  # Increase font size for footer text


# Color Schemes
QUANTA_COLORS = {
    "background": "#1e1e1e",
    "base_stroke": "#555555",
    "base_text": "#d0d0d0",
    "caption": "#b0b0b0",
    "highlight_leaf": "#ff6f61",
    "highlight_branch": "#6ac1ff",
    "highlight_enclose": "#c792ea",
    "faded_opacity": 0.4,
    "default_stroke_width": "1.2",
    "highlight_stroke_width": "2.0",
    "font_family": FONT_SANS_SERIF,
    "font_size": "11",
    "caption_font_family": FONT_SERIF,
    "caption_font_size_add": 1,
}

LIGHT_COLORS = {
    "background": "#ffffff",
    "base_stroke": "#B0B0B0",
    "base_text": "#404040",
    "caption": "#555555",
    "highlight_leaf": "#E66100",
    "highlight_branch": "#5D3A9B",
    "highlight_enclose": "#1B9E77",
    "faded_opacity": 0.30,
    "default_stroke_width": "1.5",
    "highlight_stroke_width": "2.5",
    "font_family": FONT_SANS_SERIF,
    "font_size": "11",
    "caption_font_family": FONT_SERIF,
    "caption_font_size_add": 1,
}

# Material Design 3 inspired color schemes
MATERIAL_LIGHT_COLORS = {
    "background": "#FAFAFA",  # Material light background
    "base_stroke": "#79747E",  # MD3 Neutral Variant 60
    "base_text": "#1C1B1F",  # MD3 Neutral 10
    "caption": "#49454F",  # MD3 Neutral 40
    # Highlight colors with MD3 accent colors
    "highlight_leaf": "#6750A4",  # MD3 Primary 40
    "highlight_branch": "#B93E94",  # MD3 Secondary 40
    "highlight_enclose": "#006C51",  # MD3 Tertiary 40
    # Enhanced styling
    "faded_opacity": 0.30,
    "default_stroke_width": "1.5",
    "highlight_stroke_width": "2.5",
    # Typography matching MD3
    "font_family": "'Roboto Flex', 'Helvetica Neue', Arial, sans-serif",
    "font_size": "11",
    "caption_font_family": "'Roboto Flex', 'Helvetica Neue', Arial, sans-serif",
    "caption_font_size_add": 1,
    # Material Design 3 elevation and shadows
    "use_elevation": True,
    "branch_roundness": 3,  # Corner radius for paths
    "node_marker_size": 6,  # Size of node markers,
}

MATERIAL_DARK_COLORS = {
    "background": "#141218",  # Material dark background
    "base_stroke": "#CAC4D0",  # MD3 Neutral Variant 70
    "base_text": "#E6E0E9",  # MD3 Neutral 90
    "caption": "#CAC4D0",  # MD3 Neutral Variant 70
    # Highlight colors with MD3 accent colors
    "highlight_leaf": "#D0BCFF",  # MD3 Primary 80
    "highlight_branch": "#EFB8C8",  # MD3 Secondary 80
    "highlight_enclose": "#7ED0B1",  # MD3 Tertiary 80
    # Enhanced styling
    "faded_opacity": 0.40,
    "default_stroke_width": "1.5",
    "highlight_stroke_width": "2.5",
    # Typography matching MD3
    "font_family": "'Roboto Flex', 'Helvetica Neue', Arial, sans-serif",
    "font_size": "11",
    "caption_font_family": "'Roboto Flex', 'Helvetica Neue', Arial, sans-serif",
    "caption_font_size_add": 1,
    # Material Design 3 elevation and shadows
    "use_elevation": True,
    "branch_roundness": 3,  # Corner radius for paths
    "node_marker_size": 3.5,  # Size of node markers
}

# Nature publication theme with Material Design 3 influence
NATURE_MD3_COLORS = {
    "background": "#FFFFFF",  # Clean white background (Nature standard)
    "base_stroke": "#666666",  # Dark gray for branches
    "base_text": "#333333",  # Near-black text for readability
    "caption": "#666666",  # Standard gray for captions
    # Highlight colors chosen for scientific clarity and colorblind accessibility
    "highlight_leaf": "#1A85FF",  # Blue (colorblind-friendly)
    "highlight_branch": "#D41159",  # Magenta (colorblind-friendly)
    "highlight_enclose": "#417505",  # Green (colorblind-friendly)
    # Additional highlight colors (colorblind-friendly palette)
    "highlight_1": "#A71D31",  # Burgundy red
    "highlight_2": "#7D4E90",  # Purple
    "highlight_3": "#FF8D3F",  # Orange
    "highlight_4": "#11805C",  # Teal
    "highlight_5": "#7A6C36",  # Olive
    # Enhanced styling for print publication with MD3 influence
    "faded_opacity": 0.25,
    "default_stroke_width": "3",
    "highlight_stroke_width": "2.0",
    # Typography combining Nature and MD3
    "font_family": "'Roboto Flex', 'Helvetica Neue', Arial, sans-serif",
    "font_size": "10",
    "caption_font_family": "'Roboto Flex', 'Helvetica Neue', Arial, sans-serif",
    "caption_font_size": "9",
    # Material Design subtle enhancements but print-friendly
    "use_elevation": False,  # No shadows for print
    "branch_roundness": 1.5,  # Subtle corner radius
    "node_marker_size": 3.0,  # Size of node markers
}

# Material Design 3 Scientific Visualization color schemes
# Based on the Material 3 design system with optimizations for scientific plots

# MD3 SCIENTIFIC COLOR SYSTEM - LIGHT
MD3_SCIENTIFIC_LIGHT = {
    "background": "#FDFCFF",  # MD3 Surface light
    "base_stroke": "#605D64",  # MD3 Neutral Variant 50
    "base_text": "#1D1B20",  # MD3 Neutral 10
    "caption": "#49454E",  # MD3 Neutral 40
    # Scientific palette with better distinction for visualizations (MD3 compatible)
    "highlight_leaf": "#6750A4",  # MD3 Primary 40
    "highlight_branch": "#A73775",  # MD3 Secondary 40
    "highlight_enclose": "#006A6A",  # MD3 Tertiary 40
    # Additional highlight colors for complex visualizations
    "highlight_1": "#B4261E",  # MD3 Error 40
    "highlight_2": "#9C4146",  # MD3 Neutral Variant 60
    "highlight_3": "#B8672C",  # MD3 Custom orange 60
    "highlight_4": "#616200",  # MD3 Custom yellow 40
    "highlight_5": "#1D6D2F",  # MD3 Custom green 40
    # Enhanced styling for print and screen
    "faded_opacity": 0.30,
    "default_stroke_width": "1.2",
    "highlight_stroke_width": "2.0",
    # MD3 Typography
    "font_family": "'Roboto Flex', 'Inter', system-ui, -apple-system, sans-serif",
    "font_size": "11",
    "caption_font_family": "'Roboto Flex', 'Inter', system-ui, -apple-system, sans-serif",
    "caption_font_size_add": 1,
    # Material Design 3 components
    "use_elevation": True,
    "branch_roundness": 2,  # Corner radius for paths
    "node_marker_size": 3.5,  # Size of node markers
    # MD3 state layer opacities
    "hover_opacity": 0.08,  # For interactive elements
    "pressed_opacity": 0.12,  # For interactive elements
    "selected_opacity": 0.16,  # For selected state
}

# MD3 SCIENTIFIC COLOR SYSTEM - DARK
MD3_SCIENTIFIC_DARK = {
    "background": "#141218",  # MD3 Surface dark
    "base_stroke": "#CAC4D0",  # MD3 Neutral Variant 70
    "base_text": "#E6E0E9",  # MD3 Neutral 90
    "caption": "#CAC4D0",  # MD3 Neutral Variant 70
    # Scientific palette with better distinction for visualizations (MD3 compatible)
    "highlight_leaf": "#D0BCFF",  # MD3 Primary 80
    "highlight_branch": "#F5B6D0",  # MD3 Secondary 80
    "highlight_enclose": "#86CECA",  # MD3 Tertiary 80
    # Additional highlight colors for complex visualizations
    "highlight_1": "#FFB4AB",  # MD3 Error 80
    "highlight_2": "#E9B9BC",  # MD3 Neutral Variant 80
    "highlight_3": "#FFBA8A",  # MD3 Custom orange 80
    "highlight_4": "#E9DE73",  # MD3 Custom yellow 80
    "highlight_5": "#85DA8F",  # MD3 Custom green 80
    # Enhanced styling for dark mode
    "faded_opacity": 0.40,
    "default_stroke_width": "1.5",
    "highlight_stroke_width": "2.5",
    # MD3 Typography - slightly larger for dark mode readability
    "font_family": "'Roboto Flex', 'Inter', system-ui, -apple-system, sans-serif",
    "font_size": "12",
    "caption_font_family": "'Roboto Flex', 'Inter', system-ui, -apple-system, sans-serif",
    "caption_font_size_add": 1,
    # Material Design 3 components
    "use_elevation": True,
    "branch_roundness": 3,  # Corner radius for paths
    "node_marker_size": 4.0,  # Size of node markers - slightly larger for dark mode
    # MD3 state layer opacities (adjusted for dark mode)
    "hover_opacity": 0.10,  # For interactive elements
    "pressed_opacity": 0.15,  # For interactive elements
    "selected_opacity": 0.20,  # For selected state
}

# MD3 SCIENTIFIC PRINT - Optimized for print publication
MD3_SCIENTIFIC_PRINT = {
    "background": "#FFFFFF",  # Pure white for print
    "base_stroke": "#444444",  # Darker for better print contrast
    "base_text": "#1A1A1A",  # Near-black for readability
    "caption": "#505050",  # Dark gray for captions
    # Print-optimized palette with CMYK-friendly colors
    "highlight_leaf": "#0063B1",  # Print-friendly blue
    "highlight_branch": "#9A0051",  # Print-friendly magenta
    "highlight_enclose": "#046A38",  # Print-friendly green
    # Additional highlight colors optimized for print
    "highlight_1": "#A30000",  # Print-friendly red
    "highlight_2": "#603F80",  # Print-friendly purple
    "highlight_3": "#CC5200",  # Print-friendly orange
    "highlight_4": "#4D6A00",  # Print-friendly olive
    "highlight_5": "#006B77",  # Print-friendly teal
    # Styling optimized for print
    "faded_opacity": 0.25,  # Lower opacity for print
    "default_stroke_width": "1.0",  # Thinner lines for print
    "highlight_stroke_width": "1.8",  # Visible but not too thick
    # Print-optimized typography
    "font_family": "'Roboto', 'Arial', 'Helvetica', sans-serif",  # Widely supported fonts
    "font_size": "9",  # Smaller for print
    "caption_font_family": "'Roboto', 'Arial', 'Helvetica', sans-serif",
    "caption_font_size_add": 1,
    # Print-friendly components
    "use_elevation": False,  # No shadows for print
    "branch_roundness": 1.5,  # Subtle corner radius
    "node_marker_size": 6.0,
}

# MD3 TREE INTERPOLATION - Specialized for phylogenetic tree transitions
MD3_TREE_INTERPOLATION = {
    "background": "#FFFFFF",  # Pure white for academic presentation
    "base_stroke": "#333333",  # Dark gray for standard branches
    "base_text": "#1D1B20",    # MD3 Neutral 10 for text
    "caption": "#49454F",      # MD3 Neutral 40 for captions
    
    # Required keys for rendering functions
    "highlight_leaf": "#6750A4",  # MD3 Primary 40 - Purple
    "highlight_branch": "#6750A4", # Same as common_splits
    "highlight_enclose": "#006C51", # MD3 Tertiary container
    
    # Primary semantic colors with mathematical meaning
    "common_splits": "#6750A4",  # MD3 Primary 40 - Purple (S_T₁ ∩ S_T₂)
    "disappearing_splits": "#B3261E",  # MD3 Error 40 - Red (S_T₁ \ S_T₂)
    "appearing_splits": "#146C2E",     # MD3 Custom Green - (S_T₂ \ S_T₁)
    "zero_length": "#79747E",        # MD3 Neutral Variant 60 - Branches with l=0
    "adjusted_length": "#006A6A",     # MD3 Tertiary 40 - Branches with adjusted length
    
    # Tonal variants for the interpolation steps
    "primary": {
        "20": "#D0BCFF",  # Lighter purple for subtle elements
        "40": "#6750A4",  # Standard purple for primary highlights
        "80": "#4F378B",  # Darker purple for emphasis
    },
    "secondary": {
        "20": "#F2B8B5",  # Light red for subtle elements
        "40": "#B3261E",  # Standard red for disappearing elements
        "80": "#8C1D18",  # Dark red for strong emphasis
    },
    "tertiary": {
        "20": "#A6D7A8",  # Light green for subtle elements
        "40": "#146C2E",  # Standard green for appearing elements
        "80": "#0C4F22",  # Dark green for strong emphasis
    },
    
    # Blob and highlight properties with mathematical meaning
    "blob_opacity": "0.35",        # Subtle blob effect for print
    "faded_opacity": "0.25",       # Fade for nonessential elements
    "default_stroke_width": "1.0",  # Standard stroke width
    "highlight_stroke_width": "1.8", # Emphasized stroke width
    
    # Typography optimized for mathematical presentation
    "font_family": "'Roboto', 'Arial', 'Helvetica', sans-serif",  # Widely supported fonts
    "font_size": "9",  # Base font size for print
    "caption_font_family": "'Roboto', 'Arial', 'Helvetica', sans-serif",
    "caption_font_size_add": 1,
    
    # Print-friendly components
    "use_elevation": False,  # No shadows for print
    "branch_roundness": 1.5,  # Subtle corner radius
    "node_marker_size": 3.0,  # Optimal size for print
}

# MD3 STORY FLOW - Specialized for visual storytelling of phylogenetic transitions
MD3_STORY_FLOW = {
    "background": "#FFFFFF",  # Clean white background for print
    "base_stroke": "#616161",  # Darker base stroke for better contrast
    "base_text": "#1D1B20",    # MD3 Neutral 10 for text
    "caption": "#49454F",      # MD3 Neutral 40 for captions
    
    # Enhanced semantic colors with stronger storytelling through color
    "highlight_leaf": "#6750A4",  # MD3 Primary 40 - Purple 
    "highlight_branch": "#6750A4", # Same as common_splits
    "highlight_enclose": "#006C51", # MD3 Tertiary container
    
    # Color meanings with mathematical and narrative purpose
    "common_splits": "#6750A4",     # MD3 Primary 40 - Purple (S_T₁ ∩ S_T₂) - Core story element
    "disappearing_splits": "#B3261E", # MD3 Error 40 - Red (S_T₁ \ S_T₂) - What's being removed
    "appearing_splits": "#006E1C",   # Green - (S_T₂ \ S_T₁) - What's being added
    "zero_length": "#79747E",        # MD3 Neutral Variant 60 - Less important elements
    "adjusted_length": "#0160A7",    # Blue - Branches being modified - Transition element
    
    # Story flow accent colors - create visual narrative
    "original": "#6750A4",     # The starting point (purple)
    "transition": "#7D57FF",   # The process of change (vibrant purple)
    "consensus": "#4A59C6",    # The shared elements (blue-purple)
    "final": "#0160A7",        # The destination (blue)
    
    # Visual enhancement properties
    "blob_opacity": "0.28",        # Subtle blob effect 
    "faded_opacity": "0.25",       # Fade for nonessential elements
    "default_stroke_width": "1.2",  # Standard stroke width
    "highlight_stroke_width": "2.2", # Emphasized stroke width
    
    # Typography optimized for storytelling
    "font_family": "'Roboto', 'Inter', system-ui, -apple-system, sans-serif",
    "font_size": "11",  # Base font size
    "leaf_font_size": "14", # Slightly larger for key elements
    "caption_font_family": "'Roboto', 'Inter', system-ui, -apple-system, sans-serif",
    "caption_font_size": "13",
    
    # Visual components
    "use_elevation": True,  # Add subtle shadows for depth
    "branch_roundness": 2.5,  # Slightly rounded corners
    "node_marker_size": 3.5,  # Optimal size for visibility
    
    # Animation properties
    "hover_opacity": 0.08,
    "focus_blur_radius": 3.0,
    "focus_opacity": 0.92
}

# Add to the MD3_TREE_INTERPOLATION dictionary
MD3_TREE_INTERPOLATION.update({
    "text_color": "#FFFFFF",  # White text
    "label_color": "#FFFFFF"  # White labels
})

# Publication-inspired color scheme (deep blue, coral, cyan, purple) for print
PUB_DEEP_CORAL_PRINT = {
    "background": "#FFFFFF",         # White for print
    "base_stroke": "#18304b",        # Deep blue for standard branches
    "base_text": "#222222",          # Dark gray for text
    "caption": "#7fd8f6",            # Soft blue for captions
    "highlight_leaf": "#ff5e5b",     # Bright coral for highlighted leaves
    "highlight_branch": "#ff6f61",   # Coral for highlighted branches
    "highlight_enclose": "#a16ae8",  # Muted purple for enclosures
    "faded_opacity": 0.35,
    "default_stroke_width": "1.5",
    "highlight_stroke_width": "2.8",
    "font_family": FONT_SANS_SERIF,
    "font_size": "12",
    "caption_font_family": FONT_SERIF,
    "caption_font_size_add": 1,
    "use_elevation": False,           # No shadows for print
    "branch_roundness": 2.5,
    "node_marker_size": 4.0,
}