import logging
from typing import List, Dict
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_robinson_foulds_trajectory(
    pairwise_distances_list: List[List[float]],
    method_names: List[str],
    robinson_foulds_data: Dict[str, List[float]],
) -> go.Figure:
    """
    Create interactive stacked plots: optimization distances on top, Robinson-Foulds distances underneath for each method.

    Parameters
    ----------
    pairwise_distances_list : List[List[float]]
        Each element is a list of distances for a particular method.
    method_names : List[str]
        Names corresponding to each method.
    robinson_foulds_data : Dict[str, List[float]]
        Dictionary mapping method names to their Robinson-Foulds distance lists
    """
    n_methods = len(method_names)

    # Create subplot titles for stacked layout
    subplot_titles: List[str] = []
    for method_name in method_names:
        subplot_titles.extend(
            [
                f"{method_name} - Optimization Distances",
                f"{method_name} - Robinson-Foulds Distances",
            ]
        )

    # Create stacked subplots: 2 rows per method
    fig = make_subplots(
        rows=n_methods * 2,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.03,
        shared_xaxes=True,
    )

    for idx, method_name in enumerate(method_names):
        # Get data
        distances: np.ndarray = np.array(pairwise_distances_list[idx], dtype=float)
        x_pos: np.ndarray = np.arange(len(distances))
        rf_distances: List[float] = robinson_foulds_data.get(method_name, [])

        # Top subplot: Optimization distances
        opt_row = idx * 2 + 1

        # Add filled area for optimization distances
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=distances,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(0, 114, 178, 0.3)",
                line=dict(color="rgba(0, 114, 178, 0.8)", width=2),
                name=f"{method_name} - Opt",
                hovertemplate="<b>Tree Pair: %{x}</b><br>Optimization Distance: %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=opt_row,
            col=1,
        )

        # Bottom subplot: Robinson-Foulds distances
        rf_row = idx * 2 + 2

        if rf_distances and len(rf_distances) == len(distances):
            rf_array: np.ndarray = np.array(rf_distances)

            # Add filled area for Robinson-Foulds distances
            fig.add_trace(
                go.Scatter(
                    x=x_pos,
                    y=rf_array,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(204, 121, 167, 0.3)",
                    line=dict(color="rgba(204, 121, 167, 0.8)", width=2),
                    name=f"{method_name} - RF",
                    hovertemplate="<b>Tree Pair: %{x}</b><br>Robinson-Foulds Distance: %{y:.4f}<extra></extra>",
                    showlegend=False,
                ),
                row=rf_row,
                col=1,
            )
        else:
            # Add text annotation for missing data
            fig.add_annotation(
                text="No Robinson-Foulds data available",
                x=0.5,
                y=0.5,
                xref=f"x{rf_row}",
                yref=f"y{rf_row}",
                showarrow=False,
                font=dict(size=12, color="gray"),
                row=rf_row,
                col=1,
            )

        # Update y-axis titles and colors
        fig.update_yaxes(
            title_text="Optimization Distance",
            title_font_color="rgba(0, 114, 178, 0.8)",
            tickfont_color="rgba(0, 114, 178, 0.8)",
            row=opt_row,
            col=1,
        )

        if rf_distances:
            fig.update_yaxes(
                title_text="Robinson-Foulds Distance",
                title_font_color="rgba(204, 121, 167, 0.8)",
                tickfont_color="rgba(204, 121, 167, 0.8)",
                row=rf_row,
                col=1,
            )

        # Add total annotations using data coordinates
        total_opt: float = float(distances.sum())
        fig.add_annotation(
            text=f"Total: {total_opt:.4f}",
            x=len(distances) * 0.02,  # Position at 2% of x-axis length
            y=distances.max() * 0.95,  # Position at 95% of y-axis max
            xanchor="left",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(0, 114, 178, 0.2)",
            bordercolor="rgba(0, 114, 178, 0.5)",
            borderwidth=1,
            font=dict(size=10),
            row=opt_row,
            col=1,
        )

        if rf_distances:
            total_rf: float = sum(rf_distances)
            rf_array: np.ndarray = np.array(rf_distances)
            fig.add_annotation(
                text=f"Total: {total_rf:.4f}",
                x=len(rf_distances) * 0.02,  # Position at 2% of x-axis length
                y=rf_array.max() * 0.95,  # Position at 95% of y-axis max
                xanchor="left",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(204, 121, 167, 0.2)",
                bordercolor="rgba(204, 121, 167, 0.5)",
                borderwidth=1,
                font=dict(size=10),
                row=rf_row,
                col=1,
            )

    # Update x-axis for the last subplot only
    fig.update_xaxes(title_text="Tree Pair Index", row=n_methods * 2, col=1)

    # Update layout
    fig.update_layout(
        title=dict(
            text="Interactive Distance Trajectories: Optimization vs Robinson-Foulds",
            x=0.5,
            font=dict(size=16),
        ),
        height=min(300 * n_methods * 2, 2400),  # Notebook-friendly height
        width=1000,
        template="plotly_white",
        showlegend=False,
    )

    # Add grid to all subplots
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")

    fig.show()
    return fig
