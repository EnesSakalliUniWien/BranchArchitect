import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

def draw_matrix_panel(
    matrix,
    row_lbl,
    col_lbl,
    bg="#F7F7F7",
    grid="#AAAAAA",
    txt="#333333",
    font="DejaVu Sans",
    padding=6,
    y_gap=22,
    x_char=6,
    ax=None,
    entry_color_func=None,
    width=None,
    height=None,
):
    """
    Draws a candidate matrix with compact size and colored entries based on their origin.
    If ax is provided, draws into that axes (for subplot/panel use) and does NOT create a new figure.
    If ax is None, creates a new figure and axes.
    """
    n_r, n_c = len(matrix), len(matrix[0])
    col_char_len = [
        max(
            [len(str(matrix[r][c])) for r in range(n_r)] +
            ([len(str(col_lbl[c]))] if c < len(col_lbl) else [0])
        )
        for c in range(n_c)
    ]
    col_w = [x_char * n for n in col_char_len]

    # Only create a new figure if ax is None
    if ax is None:
        if width is not None and height is not None:
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig_w = (sum(col_w) + 30) / 160
            fig_h = (n_r * y_gap + 30) / 160
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=bg)
    else:
        fig = None  # No new figure created

    ax.set_facecolor(bg)
    ax.set_xlim(0, sum(col_w))
    ax.set_ylim(0, n_r * y_gap)
    ax.axis("off")

    def col_x(c):
        return sum(col_w[:c])

    for r in range(n_r):
        for c in range(n_c):
            x0 = col_x(c)
            y0 = n_r * y_gap - (r + 1) * y_gap
            entry = matrix[r][c]
            if entry_color_func is not None:
                cell_color = entry_color_func(entry, r, c)
            else:
                cell_color = bg
            ax.add_patch(
                Rectangle((x0, y0), col_w[c], y_gap, facecolor=cell_color, edgecolor=grid, lw=1)
            )
            ax.text(
                x0 + col_w[c] / 2,
                y0 + y_gap / 2,
                str(entry),
                ha="center",
                va="center",
                color=txt,
                fontproperties=FontProperties(family=font, size=9),
            )

    header_font = FontProperties(family=font, size=10, weight="bold")
    for c, label in enumerate(col_lbl):
        x = col_x(c) + col_w[c] / 2
        ax.text(
            x,
            n_r * y_gap + 10,
            label,
            ha="center",
            va="bottom",
            color=txt,
            fontproperties=header_font,
        )

    for r, label in enumerate(row_lbl):
        y = n_r * y_gap - r * y_gap - y_gap / 2
        ax.text(
            -8,
            y,
            label,
            ha="right",
            va="center",
            color=txt,
            fontproperties=header_font,
        )

    # Only call plt.tight_layout() and return fig if we created a new figure
    if fig is not None:
        plt.tight_layout()
        return fig, ax
    # If ax was provided, do not return a new figure
    return ax