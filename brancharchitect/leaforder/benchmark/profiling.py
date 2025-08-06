"""
Profiling module for benchmark utilities.

This module contains all profiling-related functionality including
performance profiling, visualization, and analysis.
"""

import logging
import cProfile
import pstats
import pandas as pd
import plotly.express as px
from typing import List, Optional, Dict, Any

from .config import PROFILING_CONFIG, DEFAULT_MIN_TIME_PERCENT


def configure_logging():
    """Configure logging for profiling operations."""
    logging.basicConfig(
        level=getattr(logging, PROFILING_CONFIG["log_level"]),
        format=PROFILING_CONFIG["log_format"]
    )


def create_profile_dataframe(
    stats: pstats.Stats,
    min_time_percent: float = DEFAULT_MIN_TIME_PERCENT,
    focus_paths: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a DataFrame from profiling stats.
    
    Parameters
    ----------
    stats : pstats.Stats
        The profiling statistics object
    min_time_percent : float
        Minimum time percentage to include in results
    focus_paths : Optional[List[str]]
        List of path patterns to focus on
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing profiling data
    """
    total_time = stats.total_tt
    profile_data = []

    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        time_percent = (tt / total_time) * 100
        time_per_call = tt / cc if cc > 0 else 0
        
        if time_percent < min_time_percent:
            continue
            
        if focus_paths:
            if not any(path in func[0] for path in focus_paths):
                continue
                
        module_parts = func[0].split("/")
        short_module = module_parts[-1] if module_parts else func[0]
        
        profile_data.append({
            "Function": f"{func[2]} ({short_module}:{func[1]})",
            "Module": short_module,
            "Line": func[1],
            "Name": func[2],
            "Calls": cc,
            "Total Time (s)": tt,
            "Time/Call (ms)": time_per_call * 1000,
            "Cumulative (s)": ct,
            "Time %": time_percent,
            "Callers": len(callers),
        })

    return pd.DataFrame(profile_data).sort_values("Total Time (s)", ascending=True)


def create_profiling_visualizations(df: pd.DataFrame) -> List[Any]:
    """
    Create profiling visualization figures.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing profiling data
        
    Returns
    -------
    List[Any]
        List of plotly figures
    """
    figs = []

    # Time breakdown bar chart
    fig1 = px.bar(
        df,
        y="Function",
        x=["Total Time (s)", "Cumulative (s)"],
        orientation="h",
        title="Time Breakdown by Function",
        barmode="overlay",
        opacity=0.7,
        template="plotly_dark",
    )
    fig1.update_layout(
        plot_bgcolor="#1f2630",
        paper_bgcolor="#1f2630",
        font={"color": "#ffffff"},
        height=min(800, max(400, len(df) * 20)),  # Notebook-friendly height
    )
    figs.append(fig1)

    # Time per call scatter plot
    fig2 = px.scatter(
        df,
        x="Calls",
        y="Time/Call (ms)",
        size="Total Time (s)",
        hover_data=["Function"],
        color="Time %",
        color_continuous_scale="Viridis",
        title="Time per Call vs Number of Calls",
        template="plotly_dark",
    )
    fig2.update_layout(
        plot_bgcolor="#1f2630", 
        paper_bgcolor="#1f2630", 
        font={"color": "#ffffff"},
        height=600  # Notebook-friendly height
    )
    figs.append(fig2)

    # Module-function heatmap
    top_n = min(20, len(df))
    fig3 = px.density_heatmap(
        df.head(top_n),
        x="Module",
        y="Function",
        z="Time %",
        title="Module-Function Time Distribution (Top 20)",
        template="plotly_dark",
        color_continuous_scale="Viridis",
    )
    fig3.update_layout(
        xaxis_tickangle=45,
        plot_bgcolor="#1f2630",
        paper_bgcolor="#1f2630",
        font={"color": "#ffffff"},
        height=600  # Notebook-friendly height
    )
    figs.append(fig3)

    return figs


def print_profile_summary(df: pd.DataFrame, total_time: float):
    """
    Print a summary of profiling results.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing profiling data
    total_time : float
        Total profiling time
    """
    print("\nProfile Summary:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Number of unique functions: {len(df)}")
    print("\nTop 5 time-consuming functions:")
    print(
        df.nlargest(5, "Total Time (s)")[
            ["Function", "Total Time (s)", "Time %", "Calls"]
        ].to_string()
    )


def run_profiler(
    target_function,
    *args,
    min_time_percent: float = DEFAULT_MIN_TIME_PERCENT,
    focus_paths: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Run profiler on a target function and return results.
    
    Parameters
    ----------
    target_function : callable
        Function to profile
    *args : tuple
        Arguments to pass to target function
    min_time_percent : float
        Minimum time percentage to include in results
    focus_paths : Optional[List[str]]
        List of path patterns to focus on
    **kwargs : dict
        Keyword arguments to pass to target function
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing profiling results
    """
    configure_logging()
    profiler = cProfile.Profile()
    
    try:
        logging.info("Starting profiling...")
        profiler.enable()
        target_function(*args, **kwargs)
        profiler.disable()
        logging.info("Profiling completed successfully.")
    except Exception as e:
        profiler.disable()
        logging.error(f"An error occurred during profiling: {e}")
        raise e

    stats = pstats.Stats(profiler)
    df = create_profile_dataframe(stats, min_time_percent, focus_paths)
    
    # Create and show visualizations
    figs = create_profiling_visualizations(df)
    for fig in figs:
        fig.show()
    
    # Print summary
    print_profile_summary(df, stats.total_tt)
    
    return df