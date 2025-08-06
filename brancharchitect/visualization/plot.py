import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist
import os

# Check for optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive visualizations will use matplotlib.")

try:
    from ete3 import Tree, TreeStyle, NodeStyle, TextFace, add_face_to_node
    ETE3_AVAILABLE = True
except ImportError:
    ETE3_AVAILABLE = False
    print("Warning: ete3 not available. Tree visualizations will use matplotlib or plotly.")

def plot_tree_ete3(newick_str, clusters, colors, title="Phylogenetic Tree", output_file=None):
    """Create an ETE3 visualization of a phylogenetic tree."""
    if not ETE3_AVAILABLE:
        print("ETE3 not available. Cannot plot tree with ETE3.")
        return None

    try:
        t = Tree(newick_str)

        # Create a default tree style
        ts = TreeStyle()
        ts.show_leaf_name = True
        ts.show_branch_length = False
        ts.show_branch_support = False
        ts.title.add_face(TextFace(title, fsize=20), column=0)

        # Color leaf nodes based on cluster
        for leaf in t.get_leaves():
            node_style = NodeStyle()
            # Extract point index from leaf name (e.g., "P0" -> 0)
            try:
                point_index = int(leaf.name.replace("P", ""))
                if 0 <= point_index < len(clusters):
                    cluster_id = clusters[point_index]
                    # ETE3 expects RGB tuples for colors
                    rgb_color = tuple(int(c * 255) for c in colors[cluster_id][:3])
                    node_style["fgcolor"] = "#%02x%02x%02x" % rgb_color
                    node_style["size"] = 10
                    leaf.set_style(node_style)
                    # Add cluster name as a face
                    leaf.add_face(TextFace(f" (C{cluster_id})", fsize=8, fgcolor="#%02x%02x%02x" % rgb_color), column=1)
            except ValueError:
                # Handle cases where leaf name is not in expected format
                pass

        if output_file:
            t.render(output_file, w=400, h=400, tree_style=ts)
            return output_file
        else:
            # For interactive display in notebooks, render to a temporary file
            # and then display it. This is a workaround as ETE3's show()
            # might not work directly in all notebook environments.
            temp_file = "temp_ete3_tree.png"
            t.render(temp_file, w=400, h=400, tree_style=ts)
            return temp_file
    except Exception as e:
        print(f"Error plotting tree with ETE3: {e}")
        return None


def plot_tree_plotly(newick_str, clusters, colors, title="Phylogenetic Tree"):
    """Create a plotly visualization of a phylogenetic tree with proper structure."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Cannot create interactive tree plot.")
        return None
    
    try:
        # Import the needed libraries
        try:
            from skbio import TreeNode
        except ImportError:
            print("Warning: scikit-bio not available. Using simplified tree visualization.")
            return plot_tree_plotly_simple(newick_str, clusters, colors, title)
        
        # Parse the newick tree structure
        try:
            from io import StringIO
            tree = TreeNode.read(StringIO(newick_str.strip()), format="newick")
        except Exception as e:
            print(f"Warning: Could not parse tree structure: {e}. Using simplified layout.")
            return plot_tree_plotly_simple(newick_str, clusters, colors, title)
        
        fig = go.Figure()
        
        # Calculate node positions using a proper tree layout
        leaves = [node.name for node in tree.tips() if node.name]
        n_leaves = len(leaves)
        
        if n_leaves == 0:
            return None
        
        # Position leaves evenly along the bottom
        leaf_positions = {}
        for i, leaf in enumerate(leaves):
            x = i / max(1, n_leaves - 1) if n_leaves > 1 else 0.5
            leaf_positions[leaf] = (x, 0)
        
        # Calculate internal node positions
        node_positions = {}
        
        def assign_positions(node, depth=0):
            if node.is_tip():
                if node.name in leaf_positions:
                    return leaf_positions[node.name]
                else:
                    return (0.5, 0)  # fallback position
            
            # Get positions of children
            child_positions = [assign_positions(child, depth + 1) for child in node.children]
            if not child_positions:
                return (0.5, depth * 0.2)
            
            # Position internal node
            child_x = [pos[0] for pos in child_positions]
            x = np.mean(child_x)
            y = depth * 0.2
            
            node_positions[id(node)] = (x, y)
            return (x, y)
        
        # Calculate all positions
        assign_positions(tree)
        
        # Draw tree edges
        def draw_edges(node):
            if node.is_tip():
                return
            
            node_pos = node_positions.get(id(node), (0.5, 0))
            for child in node.children:
                if child.is_tip():
                    child_pos = leaf_positions.get(child.name, (0.5, 0))
                else:
                    child_pos = node_positions.get(id(child), (0.5, 0))
                
                # Draw L-shaped connection (phylogenetic style)
                fig.add_trace(go.Scatter(
                    x=[node_pos[0], child_pos[0]],
                    y=[node_pos[1], node_pos[1]],
                    mode='lines',
                    line=dict(color='darkgray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=[child_pos[0], child_pos[0]],
                    y=[node_pos[1], child_pos[1]],
                    mode='lines',
                    line=dict(color='darkgray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Recursively draw child edges
            for child in node.children:
                draw_edges(child)
        
        draw_edges(tree)
        
        # Draw leaf nodes with cluster colors
        for leaf in leaves:
            try:
                if leaf.startswith('P'):
                    point_index = int(leaf[1:])
                    if 0 <= point_index < len(clusters):
                        cluster_id = clusters[point_index]
                        if isinstance(colors[cluster_id], str):
                            color = colors[cluster_id]
                        else:
                            # Convert RGB tuple to plotly color
                            rgb = colors[cluster_id]
                            color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                    else:
                        color = 'gray'
                else:
                    color = 'gray'
            except:
                color = 'gray'
            
            pos = leaf_positions.get(leaf, (0.5, 0))
            fig.add_trace(go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode='markers+text',
                marker=dict(size=15, color=color, line=dict(width=2, color='black')),
                text=[leaf],
                textposition='bottom center',
                textfont=dict(size=10, color='black'),
                showlegend=False,
                hoverinfo='text',
                hovertext=f'Leaf: {leaf} (Cluster {clusters[int(leaf[1:])] if leaf.startswith("P") and int(leaf[1:]) < len(clusters) else "?"})'
            ))
        
        # Draw internal nodes
        for node_id, pos in node_positions.items():
            fig.add_trace(go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode='markers',
                marker=dict(size=8, color='lightgray', line=dict(width=1, color='black')),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=14)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=400,
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    except Exception as e:
        print(f"Error creating plotly tree: {e}")
        return plot_tree_plotly_simple(newick_str, clusters, colors, title)


def plot_tree_plotly_simple(newick_str, clusters, colors, title="Phylogenetic Tree"):
    """Create a simplified plotly tree visualization as fallback."""
    try:
        # Parse the newick string to extract leaf names
        import re
        leaf_pattern = r'P\d+'
        leaves = re.findall(leaf_pattern, newick_str)
        leaves = list(set(leaves))  # Remove duplicates
        
        # Create a radial layout for simplicity
        n_leaves = len(leaves)
        if n_leaves == 0:
            return None
            
        fig = go.Figure()
        
        # Create a dendogram-style layout
        if n_leaves > 1:
            # Position leaves evenly along x-axis
            x_positions = np.linspace(0, 1, n_leaves)
            
            # Plot leaves with cluster colors
            for i, leaf in enumerate(leaves):
                try:
                    point_index = int(leaf[1:])
                    if 0 <= point_index < len(clusters):
                        cluster_id = clusters[point_index]
                        if isinstance(colors[cluster_id], str):
                            color = colors[cluster_id]
                        else:
                            # Convert RGB tuple to plotly color
                            rgb = colors[cluster_id]
                            color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                    else:
                        color = 'gray'
                except:
                    color = 'gray'
                
                fig.add_trace(go.Scatter(
                    x=[x_positions[i]],
                    y=[0],
                    mode='markers+text',
                    marker=dict(size=15, color=color, line=dict(width=2, color='black')),
                    text=[leaf],
                    textposition='bottom center',
                    textfont=dict(size=10, color='black'),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f'Leaf: {leaf} (Cluster {clusters[int(leaf[1:])] if leaf.startswith("P") and int(leaf[1:]) < len(clusters) else "?"})'
                ))
            
            # Add simple connecting lines to show relationships
            for i in range(len(x_positions) - 1):
                fig.add_trace(go.Scatter(
                    x=[x_positions[i], x_positions[i+1]],
                    y=[0.1, 0.1],
                    mode='lines',
                    line=dict(color='darkgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                # Vertical lines to leaves
                fig.add_trace(go.Scatter(
                    x=[x_positions[i], x_positions[i]],
                    y=[0, 0.1],
                    mode='lines',
                    line=dict(color='darkgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Last vertical line
            fig.add_trace(go.Scatter(
                x=[x_positions[-1], x_positions[-1]],
                y=[0, 0.1],
                mode='lines',
                line=dict(color='darkgray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=14)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 0.3]),
            width=400,
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    except Exception as e:
        print(f"Error creating simple plotly tree: {e}")
        return None


def plot_nj_trees_plotly(nj_trees, clusters, colors):
    """Create interactive Plotly visualization grid of NJ trees"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Falling back to matplotlib visualization.")
        # Fallback to matplotlib version with correct parameters
        return plot_nj_trees_grid(nj_trees, n_divergence_steps=len(nj_trees)//2, clusters=clusters, colors=colors)

    try:
        actual_steps = len(nj_trees)
        # Calculate grid size to show all trees
        if actual_steps <= 4:
            n_cols = 2
            n_rows = 2
            n_trees_to_show = actual_steps
        elif actual_steps <= 6:
            n_cols = 3
            n_rows = 2
            n_trees_to_show = actual_steps
        elif actual_steps <= 9:
            n_cols = 3
            n_rows = 3
            n_trees_to_show = actual_steps
        elif actual_steps <= 12:
            n_cols = 4
            n_rows = 3
            n_trees_to_show = actual_steps
        elif actual_steps <= 16:
            n_cols = 4
            n_rows = 4
            n_trees_to_show = actual_steps
        else:
            # For very large numbers of trees, show a selection
            n_cols = 5
            n_rows = 4
            n_trees_to_show = 20
            # Show trees at regular intervals
            step_size = max(1, actual_steps // n_trees_to_show)
            tree_indices = list(range(0, actual_steps, step_size))[:n_trees_to_show]
        
        # Convert seaborn colors to plotly colors if needed
        if colors and hasattr(colors[0], '__len__') and len(colors[0]) >= 3:
            # Convert RGB tuples to plotly color strings
            plotly_colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]
        else:
            # Use colors as-is if they're already strings
            plotly_colors = colors

        # Handle tree selection for large datasets
        if actual_steps > 16:
            selected_trees = [(tree_indices[i], nj_trees[tree_indices[i]]) for i in range(n_trees_to_show)]
            subplot_titles = [f"Step {idx}" for idx, _ in selected_trees]
        else:
            selected_trees = [(i, nj_trees[i]) for i in range(n_trees_to_show)]
            subplot_titles = [f"Step {idx}" for idx, _ in selected_trees]

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "scatter"} for _ in range(n_cols)] for _ in range(n_rows)],
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )

        for i, (step_idx, tree_str) in enumerate(selected_trees):
            tree_fig = plot_tree_plotly(tree_str, clusters, plotly_colors, f"Step {step_idx}")
            if tree_fig:
                row = i // n_cols + 1
                col = i % n_cols + 1

                for trace in tree_fig.data:
                    fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            height=300 * n_rows,
            title_text="Neighbor-Joining Trees Evolution (Interactive)",
            title_font_size=20,
            showlegend=False,
        )

        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(
                    showgrid=False, zeroline=False, showticklabels=False, row=i, col=j
                )
                fig.update_yaxes(
                    showgrid=False, zeroline=False, showticklabels=False, row=i, col=j
                )

        fig.show()
        print(
            f"Displayed {n_trees_to_show} interactive trees out of {actual_steps} total trees"
        )
    except Exception as e:
        print(f"Error creating plotly NJ trees: {e}")
        # Fallback to matplotlib
        plot_nj_trees_grid(nj_trees, n_divergence_steps=len(nj_trees)//2)


def plot_3d_trajectory_plotly(full_traj, centroids, centroid_targets, clusters, modes, n_points, n_centroids, ax_lim):
    """Create interactive 3D trajectory visualization with Plotly"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Falling back to matplotlib visualization.")
        # Fallback with corrected parameters
        return plot_3d_motion_grid(full_traj, centroids, centroid_targets, clusters, modes, 
                                   n_points, n_centroids, ax_lim, n_divergence_steps=full_traj.shape[1]//2)
    
    try:
        actual_steps = full_traj.shape[1]
        colors = px.colors.qualitative.Set1[:n_centroids]
        pt_colors = [colors[c] for c in clusters]
        
        # Create animated 3D scatter plot
        frames = []
        
        for step in range(actual_steps):
            frame_data = []
            
            # Points
            for i in range(n_points):
                marker_symbol = 'circle' if modes[i] == 0 else 'x'
                frame_data.append(
                    go.Scatter3d(
                        x=[full_traj[i, step, 0]],
                        y=[full_traj[i, step, 1]],
                        z=[full_traj[i, step, 2]],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=pt_colors[i],
                            symbol=marker_symbol,
                            line=dict(color='white', width=1)
                        ),
                        name=f'P{i}',
                        text=f'Cluster {clusters[i]}',
                        hoverinfo='text+name',
                        showlegend=(step == 0)
                    )
                )
            
            # Centroids
            for j, centroid in enumerate(centroids):
                frame_data.append(
                    go.Scatter3d(
                        x=[centroid[0]],
                        y=[centroid[1]],
                        z=[centroid[2]],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=colors[j],
                            symbol='diamond',
                            line=dict(color='black', width=2)
                        ),
                        name=f'Centroid {j}',
                        showlegend=(step == 0)
                    )
                )
            
            frames.append(go.Frame(data=frame_data, name=str(step)))
        
        # Initial frame
        fig = go.Figure(data=frames[0].data, frames=frames)
        
        # Add play/pause buttons
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-ax_lim, ax_lim]),
                yaxis=dict(range=[-ax_lim, ax_lim]),
                zaxis=dict(range=[-ax_lim, ax_lim]),
                aspectmode='cube'
            ),
            updatemenus=[{'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{'steps': [
                    {
                        'args': [[str(k)], {'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }],
                        'label': f'Step {k}',
                        'method': 'animate'
                    }
                    for k in range(actual_steps)
                ],
                'active': 0,
                'y': 0,
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }],
            title='3D Trajectory Animation',
            height=800
        )
        
        fig.show()
        print("Displayed interactive 3D trajectory animation")
    except Exception as e:
        print(f"Error creating plotly 3D trajectory: {e}")
        # Fallback to matplotlib
        plot_3d_motion_grid(full_traj, centroids, centroid_targets, clusters, modes, 
                            n_points, n_centroids, ax_lim, n_divergence_steps=full_traj.shape[1]//2)

def enhanced_cube_frame(ax, lim, alpha=0.3):
    """Draw enhanced 3D cube frame."""
    # Main cube edges
    for s in (-lim, lim):
        ax.plot(
            [s, s],
            [-lim, -lim],
            [-lim, lim],
            color="lightgray",
            ls="-",
            lw=1.2,
            alpha=alpha,
        )
        ax.plot(
            [s, s],
            [lim, lim],
            [-lim, lim],
            color="lightgray",
            ls="-",
            lw=1.2,
            alpha=alpha,
        )
        ax.plot(
            [-lim, lim],
            [s, s],
            [-lim, -lim],
            color="lightgray",
            ls="-",
            lw=1.2,
            alpha=alpha,
        )
        ax.plot(
            [-lim, lim],
            [s, s],
            [lim, lim],
            color="lightgray",
            ls="-",
            lw=1.2,
            alpha=alpha,
        )

    # Grid lines for depth perception
    for coord in np.linspace(-lim, lim, 5):
        ax.plot(
            [coord, coord],
            [-lim, lim],
            [-lim, -lim],
            color="lightgray",
            ls=":",
            lw=0.5,
            alpha=alpha / 2,
        )
        ax.plot(
            [-lim, lim],
            [coord, coord],
            [-lim, -lim],
            color="lightgray",
            ls=":",
            lw=0.5,
            alpha=alpha / 2,
        )

def plot_3d_motion_grid(full_traj, centroids, centroid_targets, clusters, modes, n_points, n_centroids, ax_lim, n_divergence_steps):
    """Create enhanced 3D motion visualization grid."""
    actual_steps = full_traj.shape[1]
    colors = sns.color_palette("husl", n_centroids)
    pt_colors = [colors[c] for c in clusters]

    # Phase color gradients
    phase_colors = [
        "#f0f8ff",
        "#e6f3ff",
        "#cce7ff",
        "#b3dbff",
        "#99d6ff",
        "#80ccff",
        "#66c2ff",
        "#4db8ff",
        "#33adff",
        "#1aa3ff",
        "#0099ff",
        "#0080e6",
        "#ffeaa7",
        "#fdcb6e",
        "#e17055",
    ]

    # Calculate grid size to show representative steps
    if actual_steps <= 4:
        n_cols = 2
        n_rows = 2
        n_steps_to_show = actual_steps
        step_indices = list(range(actual_steps))
    elif actual_steps <= 6:
        n_cols = 3
        n_rows = 2
        n_steps_to_show = actual_steps
        step_indices = list(range(actual_steps))
    elif actual_steps <= 9:
        n_cols = 3
        n_rows = 3
        n_steps_to_show = actual_steps
        step_indices = list(range(actual_steps))
    else:
        # For larger datasets, show key time points
        n_cols = 4
        n_rows = 3
        n_steps_to_show = 12
        # Show beginning, middle, transition, and end
        step_size = max(1, actual_steps // n_steps_to_show)
        step_indices = list(range(0, actual_steps, step_size))[:n_steps_to_show]
    
    fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 4.5), facecolor="white")

    axes = [
        fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        for i in range(n_steps_to_show)
    ]
    transition_step = n_divergence_steps - 1

    for i, ax in enumerate(axes):
        if i >= len(step_indices):
            ax.axis('off')  # Hide extra axes
            continue
            
        step = step_indices[i]
        
        # Set phase-based background
        if step <= transition_step:
            color_idx = min(step, len(phase_colors) - 1)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = True
                pane.set_facecolor(phase_colors[color_idx])
                pane.set_alpha(0.1)

        plot_pos = full_traj[:, step, :]

        # Draw trajectory trails
        if step > 0:
            for pt_idx in range(n_points):
                prev_pos = full_traj[pt_idx, max(0, step - 4) : step + 1, :]
                if len(prev_pos) > 1:
                    alpha_values = np.linspace(0.1, 0.7, len(prev_pos))
                    for j in range(len(prev_pos) - 1):
                        ax.plot(
                            [prev_pos[j, 0], prev_pos[j + 1, 0]],
                            [prev_pos[j, 1], prev_pos[j + 1, 1]],
                            [prev_pos[j, 2], prev_pos[j + 1, 2]],
                            color=pt_colors[pt_idx],
                            alpha=alpha_values[j],
                            lw=1.5,
                        )

        # Plot current positions
        for pt_idx in range(n_points):
            marker = "o" if modes[pt_idx] == 0 else "X"
            size = 90 if modes[pt_idx] == 0 else 110
            ax.scatter(
                *plot_pos[pt_idx],
                color=pt_colors[pt_idx],
                marker=marker,
                s=size,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.9,
            )

        # Plot centroids and targets
        for j, centroid in enumerate(centroids):
            ax.scatter(
                *centroid,
                s=300,
                marker="^",
                color=colors[j],
                edgecolor="black",
                linewidth=2,
                alpha=0.8,
            )

            if step == actual_steps - 1:  # Final step connections
                target = centroid_targets[j]
                ax.plot(
                    [centroid[0], target[0]],
                    [centroid[1], target[1]],
                    [centroid[2], target[2]],
                    color=colors[j],
                    ls="--",
                    lw=2,
                    alpha=0.6,
                )
                ax.scatter(
                    *target,
                    s=200,
                    marker="s",
                    color=colors[j],
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.7,
                )

        # Origin marker
        ax.scatter(
            0, 0, 0, marker="*", color="gold", s=150, edgecolor="black", linewidth=1
        )

        # Configure axes
        enhanced_cube_frame(ax, ax_lim)
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_zlim(-ax_lim, ax_lim)

        # Title with phase information
        phase = "Divergence" if step <= transition_step else "Resolution"
        title_color = "#2980b9" if step <= transition_step else "#e67e22"
        ax.set_title(
            f"Step {step} ({phase} Phase)",
            fontsize=10,
            fontweight="bold",
            color=title_color,
            pad=10,
        )
        ax.view_init(elev=25, azim=-60)

        # Subtle grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X", fontsize=7, alpha=0.7)
        ax.set_ylabel("Y", fontsize=7, alpha=0.7)
        ax.set_zlabel("Z", fontsize=7, alpha=0.7)

    # Add legend
    if len(axes) > 0:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="Linear Motion",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="Ballistic Motion",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label="Centroids",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gold",
                markersize=12,
                label="Origin",
            ),
        ]
        axes[0].legend(
            handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1), fontsize=7
        )

    fig.suptitle(
        f"Two-Phase Simulation: {n_divergence_steps}-Step Divergence → {full_traj.shape[1] - n_divergence_steps}-Step Resolution",
        fontsize=18,
        fontweight="bold",
        y=0.96,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    # Save and show plot
    try:
        plt.savefig("3d_motion_grid.png", dpi=150, bbox_inches="tight")
        print("Saved 3D motion visualization to: 3d_motion_grid.png")
    except Exception as e:
        print(f"Warning: Could not save 3D motion grid: {e}")
    plt.show()

def plot_nj_tree_text(ax, tree_str, step, transition_step, clusters=None, colors=None):
    """Plot NJ tree as 2D visualization with cluster coloring."""
    # Use the new 2D tree visualization if clusters and colors are provided
    if clusters is not None and colors is not None:
        plot_tree_2d_cluster(ax, tree_str, clusters, colors, 
                           title="NJ Tree", step=step, transition_step=transition_step)
    else:
        # Fallback to text display
        ax.clear()
        ax.axis("off")

        # Format tree string for better readability
        display_tree = tree_str.replace("(", "\n  (").replace(")", ")\n")
        display_tree = display_tree.replace(",", ",\n    ")
        lines = display_tree.split("\n")
        
        # Clean up empty lines
        lines = [line for line in lines if line.strip()]

        # Limit lines and add color coding
        max_lines = 12  # Reduced for better visibility
        if len(lines) > max_lines:
            lines = lines[: max_lines - 1] + ["..."]

        if step <= transition_step:
            ax.set_facecolor("#f0f8ff")
            phase_label = "Divergence"
            title_color = "#2980b9"
            bg_color = "#e8f4fd"
        else:
            ax.set_facecolor("#fff5f5")
            phase_label = "Resolution"
            title_color = "#e67e22"
            bg_color = "#fdeee8"

        # Display tree text with better formatting
        tree_text = "\n".join(lines)
        ax.text(
            0.05,
            0.95,
            tree_text,
            transform=ax.transAxes,
            fontsize=8,  # Increased font size
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, alpha=0.9, edgecolor="gray"),
            wrap=True
        )

        ax.set_title(
            f"Step {step} - NJ Tree ({phase_label})",
            fontsize=12,  # Increased title size
            fontweight="bold",
            pad=15,
            color=title_color,
        )
        
        # Add tree statistics
        n_taxa = tree_str.count(',') + 1 if tree_str.count(',') > 0 else 0
        ax.text(
            0.02,
            0.02,
            f"Taxa: {n_taxa}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"),
        )

def plot_nj_trees_grid(nj_trees, n_divergence_steps, clusters=None, colors=None):
    """Create NJ tree visualization grid with cluster coloring."""
    actual_steps = len(nj_trees)
    
    # Calculate grid size to show all trees
    if actual_steps <= 4:
        n_cols = 2
        n_rows = 2
        n_trees_to_show = actual_steps
    elif actual_steps <= 6:
        n_cols = 3
        n_rows = 2
        n_trees_to_show = actual_steps
    elif actual_steps <= 9:
        n_cols = 3
        n_rows = 3
        n_trees_to_show = actual_steps
    elif actual_steps <= 12:
        n_cols = 4
        n_rows = 3
        n_trees_to_show = actual_steps
    elif actual_steps <= 16:
        n_cols = 4
        n_rows = 4
        n_trees_to_show = actual_steps
    else:
        # For very large numbers of trees, show a selection
        n_cols = 5
        n_rows = 4
        n_trees_to_show = 20
        # Show trees at regular intervals
        step_size = max(1, actual_steps // n_trees_to_show)
        tree_indices = list(range(0, actual_steps, step_size))[:n_trees_to_show]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    # Handle tree selection for large datasets
    if actual_steps > 16:
        selected_trees = [(tree_indices[i], nj_trees[tree_indices[i]]) for i in range(n_trees_to_show)]
    else:
        selected_trees = [(i, nj_trees[i]) for i in range(n_trees_to_show)]

    # Hide extra axes
    for i in range(n_trees_to_show, len(axes)):
        axes[i].axis("off")

    transition_step = n_divergence_steps - 1

    for i, (step_idx, tree_str) in enumerate(selected_trees):
        if i < len(axes):
            plot_nj_tree_text(axes[i], tree_str, step_idx, transition_step, clusters, colors)
    
    # Hide remaining axes if any
    for i in range(len(selected_trees), len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "Neighbor-Joining Trees - Phylogenetic Evolution",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    # Save and show plot
    try:
        plt.savefig("nj_trees_grid.png", dpi=150, bbox_inches="tight")
        print("Saved NJ trees visualization to: nj_trees_grid.png")
    except Exception as e:
        print(f"Warning: Could not save NJ trees grid: {e}")
    plt.show()

def plot_tree_2d_cluster(ax, newick_str, clusters, colors, title="Phylogenetic Tree", step=None, transition_step=None):
    """Create a 2D tree visualization with cluster coloring."""
    try:
        from skbio import TreeNode
        import re
        
        ax.clear()
        
        # Parse the newick tree
        try:
            from io import StringIO
            tree = TreeNode.read(StringIO(newick_str.strip()), format="newick")
        except Exception as e:
            print(f"Warning: Could not parse tree structure: {e}")
            # Fall back to simple visualization
            plot_tree_2d_simple(ax, newick_str, clusters, colors, title, step, transition_step)
            return
        
        # Extract leaf names and assign coordinates
        leaves = [node.name for node in tree.tips() if node.name]
        n_leaves = len(leaves)
        
        if n_leaves == 0:
            ax.text(0.5, 0.5, "No tree data", ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create a hierarchical layout
        # Position leaves evenly along x-axis
        leaf_positions = {}
        leaf_x = np.linspace(0, 1, n_leaves)
        
        # Assign y=0 to all leaves
        for i, leaf in enumerate(leaves):
            leaf_positions[leaf] = (leaf_x[i], 0)
        
        # Calculate internal node positions
        node_positions = {}
        
        def assign_positions(node):
            if node.is_tip():
                if node.name in leaf_positions:
                    return leaf_positions[node.name]
                else:
                    return (0.5, 0)  # fallback position
            
            # Get positions of children
            child_positions = [assign_positions(child) for child in node.children]
            if not child_positions:
                return (0.5, 0.1)
            
            child_x = [pos[0] for pos in child_positions]
            child_y = [pos[1] for pos in child_positions]
            
            # Position internal node
            x = np.mean(child_x)
            y = max(child_y) + 0.1  # Move up from children
            
            node_positions[id(node)] = (x, y)
            return (x, y)
        
        # Calculate all positions
        assign_positions(tree)
        
        # Draw tree edges
        def draw_edges(node):
            if node.is_tip():
                return
            
            node_pos = node_positions.get(id(node), (0.5, 0))
            for child in node.children:
                if child.is_tip():
                    child_pos = leaf_positions.get(child.name, (0.5, 0))
                else:
                    child_pos = node_positions.get(id(child), (0.5, 0))
                
                # Draw L-shaped connection
                ax.plot([node_pos[0], child_pos[0]], [node_pos[1], node_pos[1]], 
                       'k-', linewidth=1.5, alpha=0.7)
                ax.plot([child_pos[0], child_pos[0]], [node_pos[1], child_pos[1]], 
                       'k-', linewidth=1.5, alpha=0.7)
            
            # Recursively draw child edges
            for child in node.children:
                draw_edges(child)
        
        draw_edges(tree)
        
        # Draw leaf nodes with cluster colors
        for i, leaf in enumerate(leaves):
            try:
                if leaf and leaf.startswith('P'):
                    point_index = int(leaf[1:])
                    if 0 <= point_index < len(clusters):
                        cluster_id = clusters[point_index]
                        if isinstance(colors[cluster_id], str):
                            color = colors[cluster_id]
                        else:
                            # Convert RGB tuple to matplotlib color
                            color = colors[cluster_id]
                    else:
                        color = 'gray'
                else:
                    color = 'gray'
            except:
                color = 'gray'
            
            pos = leaf_positions.get(leaf, (0.5, 0))
            ax.scatter(pos[0], pos[1], c=[color], s=100, marker='o', 
                      edgecolors='black', linewidth=1, zorder=10)
            
            # Add cluster label to the leaf
            cluster_label = ""
            try:
                if leaf and leaf.startswith('P'):
                    point_index = int(leaf[1:])
                    if 0 <= point_index < len(clusters):
                        cluster_id = clusters[point_index]
                        cluster_label = f"{leaf}\n(C{cluster_id})"
                    else:
                        cluster_label = leaf
                else:
                    cluster_label = leaf
            except:
                cluster_label = leaf
            
            ax.text(pos[0], pos[1]-0.08, cluster_label, ha='center', va='top', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Draw internal nodes
        for node_id, pos in node_positions.items():
            ax.scatter(pos[0], pos[1], c='lightgray', s=50, marker='s', 
                      edgecolors='black', linewidth=1, zorder=5, alpha=0.7)
        
        # Set phase-based styling
        if step is not None and transition_step is not None:
            if step <= transition_step:
                ax.set_facecolor("#f0f8ff")
                phase_label = "Divergence"
                title_color = "#2980b9"
                bg_color = "#e8f4fd"
            else:
                ax.set_facecolor("#fff5f5")
                phase_label = "Resolution"
                title_color = "#e67e22"
                bg_color = "#fdeee8"
            
            title = f"{title} - Step {step} ({phase_label})"
        else:
            title_color = "black"
        
        ax.set_title(title, fontsize=12, fontweight="bold", color=title_color, pad=10)
        ax.set_xlim(-0.1, 1.1)
        max_y = max([pos[1] for pos in node_positions.values()]) if node_positions else 0.1
        ax.set_ylim(-0.2, max_y + 0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add cluster legend
        if len(set(clusters)) <= 10:  # Only show legend if not too many clusters
            unique_clusters = sorted(set(clusters))
            legend_elements = []
            for cluster_id in unique_clusters:
                if isinstance(colors[cluster_id], str):
                    color = colors[cluster_id]
                else:
                    color = colors[cluster_id]
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                              markersize=8, label=f'Cluster {cluster_id}', markeredgecolor='black')
                )
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                     framealpha=0.9, fancybox=True)
        
    except Exception as e:
        # Fallback to simplified visualization
        print(f"Warning: Tree visualization failed: {e}")
        plot_tree_2d_simple(ax, newick_str, clusters, colors, title, step, transition_step)


def plot_tree_2d_simple(ax, newick_str, clusters, colors, title="Phylogenetic Tree", step=None, transition_step=None):
    """Create a simple 2D tree visualization as fallback."""
    try:
        import re
        
        ax.clear()
        
        # Extract leaf names from newick string
        leaf_pattern = r'P\\d+'
        leaves = re.findall(leaf_pattern, newick_str)
        leaves = list(set(leaves))  # Remove duplicates
        leaves.sort()  # Sort for consistent ordering
        
        n_leaves = len(leaves)
        
        if n_leaves == 0:
            ax.text(0.5, 0.5, "No leaves found in tree", ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create a simple dendrogram layout
        x_positions = np.linspace(0.1, 0.9, n_leaves)
        y_leaf = 0.1
        y_internal = 0.3
        
        # Draw leaves with cluster colors
        for i, leaf in enumerate(leaves):
            try:
                if leaf.startswith('P'):
                    point_index = int(leaf[1:])
                    if 0 <= point_index < len(clusters):
                        cluster_id = clusters[point_index]
                        if isinstance(colors[cluster_id], str):
                            color = colors[cluster_id]
                        else:
                            # Convert RGB tuple to matplotlib color
                            color = colors[cluster_id]
                    else:
                        color = 'gray'
                else:
                    color = 'gray'
            except:
                color = 'gray'
            
            # Draw leaf node
            ax.scatter(x_positions[i], y_leaf, c=[color], s=100, marker='o', 
                      edgecolors='black', linewidth=1, zorder=10)
            
            # Add label with cluster info
            cluster_label = ""
            try:
                if leaf.startswith('P'):
                    point_index = int(leaf[1:])
                    if 0 <= point_index < len(clusters):
                        cluster_id = clusters[point_index]
                        cluster_label = f"{leaf}\n(C{cluster_id})"
                    else:
                        cluster_label = leaf
                else:
                    cluster_label = leaf
            except:
                cluster_label = leaf
            
            ax.text(x_positions[i], y_leaf - 0.05, cluster_label, ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Draw vertical line to internal level
            ax.plot([x_positions[i], x_positions[i]], [y_leaf, y_internal], 
                   'k-', linewidth=1, alpha=0.7)
        
        # Draw horizontal line connecting all leaves
        if n_leaves > 1:
            ax.plot([x_positions[0], x_positions[-1]], [y_internal, y_internal], 
                   'k-', linewidth=2, alpha=0.7)
        
        # Set phase-based styling
        if step is not None and transition_step is not None:
            if step <= transition_step:
                ax.set_facecolor("#f0f8ff")
                phase_label = "Divergence"
                title_color = "#2980b9"
            else:
                ax.set_facecolor("#fff5f5")
                phase_label = "Resolution"
                title_color = "#e67e22"
            
            title = f"{title} - Step {step} ({phase_label})"
        else:
            title_color = "black"
        
        ax.set_title(title, fontsize=12, fontweight="bold", color=title_color, pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add cluster legend
        if len(set(clusters)) <= 10:  # Only show legend if not too many clusters
            unique_clusters = sorted(set(clusters))
            legend_elements = []
            for cluster_id in unique_clusters:
                if isinstance(colors[cluster_id], str):
                    color = colors[cluster_id]
                else:
                    color = colors[cluster_id]
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                              markersize=8, label=f'Cluster {cluster_id}', markeredgecolor='black')
                )
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                     framealpha=0.9, fancybox=True)
        
    except Exception as e:
        # Ultimate fallback
        ax.clear()
        ax.text(0.5, 0.5, f"Tree visualization failed:\n{str(e)[:100]}", 
               ha='center', va='center', transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        ax.set_title(title if title else "Tree Visualization Error", fontsize=12)
        ax.axis('off')


def plot_analytics_dashboard(full_traj, clusters, nj_trees, n_centroids, n_divergence_steps):
    """Create comprehensive analytics dashboard."""
    actual_steps = full_traj.shape[1]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))

    # 1. Average pairwise distances
    avg_distances = [
        np.mean(pdist(full_traj[:, step, :])) for step in range(actual_steps)
    ]
    ax1.plot(
        range(actual_steps),
        avg_distances,
        "o-",
        linewidth=2,
        markersize=5,
        color="#3498db",
    )
    ax1.axvline(
        x=n_divergence_steps - 1,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Phase Transition",
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Average Pairwise Distance")
    ax1.set_title("Average Pairwise Distances")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Cluster compactness
    cluster_compactness = []
    for step in range(actual_steps):
        compactness = []
        for cluster_id in range(n_centroids):
            cluster_points = full_traj[clusters == cluster_id, step, :]
            if len(cluster_points) > 1:
                cluster_distances = pdist(cluster_points)
                compactness.append(np.mean(cluster_distances))
        cluster_compactness.append(np.mean(compactness) if compactness else 0)

    ax2.plot(
        range(actual_steps),
        cluster_compactness,
        "s-",
        linewidth=2,
        markersize=5,
        color="#e74c3c",
    )
    ax2.axvline(
        x=n_divergence_steps - 1,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Phase Transition",
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Average Cluster Compactness")
    ax2.set_title("Cluster Compactness Evolution")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Multifurcation estimation
    multifurcation_counts = [
        tree_str.count(",") - tree_str.count("(") + 1 for tree_str in nj_trees
    ]
    dendrogram_colors = sns.color_palette("viridis", actual_steps)
    ax3.bar(
        range(actual_steps), multifurcation_counts, color=dendrogram_colors, alpha=0.7
    )
    ax3.axvline(
        x=n_divergence_steps - 1,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Phase Transition",
    )
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Estimated Multifurcations")
    ax3.set_title("NJ Tree Resolution Progress")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend()

    # 4. Tree complexity
    tree_complexity = [
        tree_str.count("(") + tree_str.count(")") for tree_str in nj_trees
    ]
    ax4.plot(
        range(actual_steps),
        tree_complexity,
        "^-",
        linewidth=2,
        markersize=6,
        color="#9b59b6",
    )
    ax4.axvline(
        x=n_divergence_steps - 1,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Phase Transition",
    )
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Tree Complexity (Node Count)")
    ax4.set_title("NJ Tree Complexity Evolution")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.suptitle("NJ Tree Analysis Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save and show plot
    try:
        plt.savefig("analytics_dashboard.png", dpi=150, bbox_inches="tight")
        print("Saved analytics dashboard to: analytics_dashboard.png")
    except Exception as e:
        print(f"Warning: Could not save analytics dashboard: {e}")
    plt.show()