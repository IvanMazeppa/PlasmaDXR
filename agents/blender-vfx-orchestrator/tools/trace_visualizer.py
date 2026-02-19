#!/usr/bin/env python3
"""
Trace Visualizer Dashboard for Blender VFX Orchestrator

Generates an interactive HTML dashboard from trace metrics CSV files
produced by trace_analyzer.py.

Usage:
    python trace_visualizer.py                          # defaults: reads tools/metrics.csv
    python trace_visualizer.py -i path/to/metrics.csv   # custom input
    python trace_visualizer.py -o dashboard.html        # custom output
    python trace_visualizer.py --open                   # auto-open in browser
    python trace_visualizer.py --no-table               # skip data table
"""

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

# --- Dependency check ---
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required but not installed.")
    print("  pip install pandas>=2.0.0")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:
    print("ERROR: plotly is required but not installed.")
    print("  pip install plotly>=5.18.0")
    sys.exit(1)


# --- Constants ---

# Prefix map: longest prefix first for greedy matching
EFFECT_PREFIX_MAP = [
    ("kitchen_leak", "kitchen_leak"),
    ("liquid_spray", "liquid_spray"),
    ("liquid_3iter", "liquid_spray"),
    ("wine_pour", "wine_pour"),
    ("oil_pour", "oil_pour"),
    ("candle", "candle"),
    ("fuel_ignition", "fuel_ignition"),
    ("spec_first", "spec_first"),
    ("det_fallback", "det_fallback"),
    ("p0_verification", "p0_verification"),
    ("shakedown_water", "shakedown"),
    ("shakedown", "shakedown"),
    ("phase4_gate_shakedown", "shakedown"),
    ("e2e_test_verbose_water", "e2e_test"),
    ("e2e_test_verbose", "e2e_test"),
    ("e2e_test", "e2e_test"),
    ("communication_flow", "communication_flow"),
]

# VFX effects get vivid colors; test/infra get muted tones
EFFECT_COLORS = {
    "kitchen_leak": "#1f77b4",
    "liquid_spray": "#2ca02c",
    "wine_pour": "#9467bd",
    "oil_pour": "#d62728",
    "candle": "#ff7f0e",
    "fuel_ignition": "#e377c2",
    "spec_first": "#aec7e8",
    "det_fallback": "#c7c7c7",
    "p0_verification": "#bcbd22",
    "shakedown": "#9c9c9c",
    "e2e_test": "#7f7f7f",
    "communication_flow": "#d9d9d9",
}

# Agent categories for stacked bar chart
AGENT_CATEGORIES = {
    "Planning": [
        "research_agent",
        "technique_selector",
        "api_spec_agent",
        "modification_strategist",
    ],
    "Execution": [
        "script_writer",
        "executor",
        "code_writer_agent",
    ],
    "Quality": [
        "quality_analyst",
        "quality_gate",
        "quality_gate_judge",
        "learning_agent",
    ],
    "Docs": [
        "documentation_expert",
        "documentation_expert_(standalone)",
    ],
}

CATEGORY_COLORS = {
    "Planning": "#636EFA",
    "Execution": "#EF553B",
    "Quality": "#00CC96",
    "Docs": "#AB63FA",
}


# --- Data Loading ---

def infer_effect_type(trace_file: str) -> str:
    """Infer effect type from trace filename using longest-prefix match."""
    name = str(trace_file).lower()
    for prefix, effect in EFFECT_PREFIX_MAP:
        if name.startswith(prefix):
            return effect
    return "unknown"


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load CSV, filter bogus rows, add derived columns."""
    df = pd.read_csv(csv_path)

    # Filter: drop communication_flow.jsonl (bogus row)
    if "trace_file" in df.columns:
        df = df[~df["trace_file"].str.contains("communication_flow", na=False)]

    # Filter: drop zero-duration rows
    if "duration_s" in df.columns:
        df = df[df["duration_s"] > 0]

    # Infer effect_type from trace_file
    if "trace_file" in df.columns:
        df["effect_type"] = df["trace_file"].apply(infer_effect_type)

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp_dt").reset_index(drop=True)
        # Short label: date + first part of trace name
        df["short_label"] = df["trace_file"].apply(
            lambda x: Path(str(x)).stem[:30] if pd.notna(x) else "unknown"
        )

    return df


def get_agent_columns(df: pd.DataFrame, suffix: str = "_duration_s") -> list:
    """Get all agent duration columns present in the dataframe."""
    return [c for c in df.columns if c.endswith(suffix) and c != "duration_s"]


# --- Chart Builders ---

def build_agent_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Chart 1: Agent Time Breakdown — stacked horizontal bar."""
    fig = go.Figure()

    agent_cols = get_agent_columns(df, "_pct")
    if not agent_cols:
        return _empty_figure("No agent percentage columns found")

    # Build category-level aggregation per row
    for category, agents in AGENT_CATEGORIES.items():
        pct_cols = [f"{a}_pct" for a in agents if f"{a}_pct" in df.columns]
        if not pct_cols:
            continue
        values = df[pct_cols].sum(axis=1).fillna(0)
        fig.add_trace(go.Bar(
            y=df["short_label"],
            x=values,
            name=category,
            orientation="h",
            marker_color=CATEGORY_COLORS.get(category, "#888"),
            hovertemplate=(
                f"<b>{category}</b><br>"
                "%{x:.1f}% of total<br>"
                "<extra>%{y}</extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        title="Agent Time Breakdown (% of total duration)",
        xaxis_title="% of Total Duration",
        yaxis_title="",
        height=max(400, len(df) * 28),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=220),
    )
    return fig


def build_duration_trends_chart(df: pd.DataFrame) -> go.Figure:
    """Chart 2: Duration Trends — line+scatter over time."""
    fig = go.Figure()

    if "timestamp_dt" not in df.columns or "duration_s" not in df.columns:
        return _empty_figure("Missing timestamp or duration columns")

    for effect in df["effect_type"].unique():
        mask = df["effect_type"] == effect
        subset = df[mask]
        color = EFFECT_COLORS.get(effect, "#888")
        fig.add_trace(go.Scatter(
            x=subset["timestamp_dt"],
            y=subset["duration_s"],
            mode="lines+markers",
            name=effect,
            marker=dict(size=8, color=color),
            line=dict(color=color, width=2),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Duration: %{y:.0f}s<br>"
                "%{x}<br>"
                "<extra></extra>"
            ),
            text=subset["short_label"],
        ))

    fig.update_layout(
        title="Duration Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Duration (seconds)",
        height=450,
        hovermode="closest",
    )
    return fig


def build_doc_search_efficiency_chart(df: pd.DataFrame) -> go.Figure:
    """Chart 3: Doc Search Efficiency — bubble scatter."""
    if "doc_searches" not in df.columns or "duration_s" not in df.columns:
        return _empty_figure("Missing doc_searches or duration_s columns")

    fig = go.Figure()

    for effect in df["effect_type"].unique():
        mask = df["effect_type"] == effect
        subset = df[mask]
        color = EFFECT_COLORS.get(effect, "#888")
        iters = subset["iterations"].fillna(1).clip(lower=1)
        fig.add_trace(go.Scatter(
            x=subset["duration_s"],
            y=subset["doc_searches"],
            mode="markers",
            name=effect,
            marker=dict(
                size=iters * 8,
                color=color,
                opacity=0.7,
                line=dict(width=1, color="white"),
                sizemin=6,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Duration: %{x:.0f}s<br>"
                "Doc Searches: %{y}<br>"
                "Iterations: %{customdata}<br>"
                "<extra></extra>"
            ),
            text=subset["short_label"],
            customdata=subset["iterations"].fillna(1).astype(int),
        ))

    fig.update_layout(
        title="Doc Search Efficiency (bubble size = iterations)",
        xaxis_title="Duration (seconds)",
        yaxis_title="Doc Searches",
        height=450,
        hovermode="closest",
    )
    return fig


def build_iteration_cost_chart(df: pd.DataFrame) -> go.Figure:
    """Chart 4: Iteration Cost — vertical bar sorted descending."""
    if "seconds_per_iteration" not in df.columns:
        return _empty_figure("Missing seconds_per_iteration column")

    sorted_df = df.dropna(subset=["seconds_per_iteration"]).sort_values(
        "seconds_per_iteration", ascending=False
    )
    if sorted_df.empty:
        return _empty_figure("No iteration cost data")

    colors = [EFFECT_COLORS.get(e, "#888") for e in sorted_df["effect_type"]]

    fig = go.Figure(go.Bar(
        x=sorted_df["short_label"],
        y=sorted_df["seconds_per_iteration"],
        marker_color=colors,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Seconds/Iteration: %{y:.1f}s<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Iteration Cost (seconds per iteration, sorted descending)",
        xaxis_title="",
        yaxis_title="Seconds per Iteration",
        height=450,
        xaxis=dict(tickangle=-45),
        margin=dict(b=150),
    )
    return fig


def build_agent_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Chart 5: Agent Duration Distribution — box plots with points."""
    agent_cols = get_agent_columns(df, "_duration_s")
    if not agent_cols:
        return _empty_figure("No agent duration columns found")

    fig = go.Figure()

    # Sort agents by median duration (descending)
    medians = {}
    for col in agent_cols:
        valid = df[col].dropna()
        if len(valid) > 0:
            medians[col] = valid.median()
    sorted_cols = sorted(medians, key=lambda c: medians[c], reverse=True)

    for col in sorted_cols:
        label = col.replace("_duration_s", "").replace("_", " ").title()
        valid = df[col].dropna()
        if len(valid) < 2:
            continue
        fig.add_trace(go.Box(
            y=valid,
            name=label,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.5,
            marker=dict(size=5, opacity=0.6),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Duration: %{y:.1f}s<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Agent Duration Distribution (box + all points)",
        yaxis_title="Duration (seconds)",
        height=500,
        showlegend=False,
    )
    return fig


def build_effect_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Chart 6: Effect Type Comparison — grouped bar with dual Y-axis."""
    if "effect_type" not in df.columns:
        return _empty_figure("No effect_type column")

    # Only include effect types with 2+ runs
    counts = df["effect_type"].value_counts()
    multi = counts[counts >= 2].index.tolist()
    if not multi:
        # Fall back to all if none have 2+
        multi = counts.index.tolist()

    fdf = df[df["effect_type"].isin(multi)]
    grouped = fdf.groupby("effect_type").agg(
        mean_duration=("duration_s", "mean"),
        mean_iterations=("iterations", "mean"),
        mean_doc_searches=("doc_searches", "mean"),
        run_count=("duration_s", "count"),
    ).reset_index()

    grouped = grouped.sort_values("mean_duration", ascending=False)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors = [EFFECT_COLORS.get(e, "#888") for e in grouped["effect_type"]]

    fig.add_trace(
        go.Bar(
            x=grouped["effect_type"],
            y=grouped["mean_duration"],
            name="Mean Duration (s)",
            marker_color=colors,
            opacity=0.8,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean Duration: %{y:.0f}s<br>"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["effect_type"],
            y=grouped["mean_iterations"],
            name="Mean Iterations",
            mode="lines+markers",
            marker=dict(size=10, symbol="diamond", color="#FF6692"),
            line=dict(color="#FF6692", width=2),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean Iterations: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["effect_type"],
            y=grouped["mean_doc_searches"],
            name="Mean Doc Searches",
            mode="lines+markers",
            marker=dict(size=10, symbol="square", color="#19D3F3"),
            line=dict(color="#19D3F3", width=2),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean Doc Searches: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Effect Type Comparison (min 2 runs)",
        height=500,
        xaxis=dict(tickangle=-30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Duration (seconds)", secondary_y=False)
    fig.update_yaxes(title_text="Count", secondary_y=True)

    return fig


def _empty_figure(message: str) -> go.Figure:
    """Return a placeholder figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#999"),
    )
    fig.update_layout(height=200)
    return fig


# --- HTML Assembly ---

def build_summary_stats(df: pd.DataFrame) -> str:
    """Build summary stats HTML box."""
    total_runs = len(df)
    if "timestamp_dt" in df.columns:
        valid_ts = df["timestamp_dt"].dropna()
        if len(valid_ts) > 0:
            date_range = f"{valid_ts.min().strftime('%Y-%m-%d')} to {valid_ts.max().strftime('%Y-%m-%d')}"
        else:
            date_range = "N/A"
    else:
        date_range = "N/A"

    mean_dur = df["duration_s"].mean() if "duration_s" in df.columns else 0
    median_dur = df["duration_s"].median() if "duration_s" in df.columns else 0
    total_dur = df["duration_s"].sum() if "duration_s" in df.columns else 0
    effect_types = df["effect_type"].nunique() if "effect_type" in df.columns else 0

    return f"""
    <div style="background:#1e1e2e;border:1px solid #444;border-radius:8px;padding:20px;margin:20px 0;
                display:flex;flex-wrap:wrap;gap:30px;justify-content:center;">
        <div style="text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#89b4fa;">{total_runs}</div>
            <div style="color:#a6adc8;font-size:13px;">Total Runs</div>
        </div>
        <div style="text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#a6e3a1;">{effect_types}</div>
            <div style="color:#a6adc8;font-size:13px;">Effect Types</div>
        </div>
        <div style="text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#f9e2af;">{mean_dur:.0f}s</div>
            <div style="color:#a6adc8;font-size:13px;">Mean Duration</div>
        </div>
        <div style="text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#cba6f7;">{median_dur:.0f}s</div>
            <div style="color:#a6adc8;font-size:13px;">Median Duration</div>
        </div>
        <div style="text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#f38ba8;">{total_dur/3600:.1f}h</div>
            <div style="color:#a6adc8;font-size:13px;">Total Time</div>
        </div>
        <div style="text-align:center;">
            <div style="font-size:14px;font-weight:bold;color:#94e2d5;">{date_range}</div>
            <div style="color:#a6adc8;font-size:13px;">Date Range</div>
        </div>
    </div>
    """


def build_data_table(df: pd.DataFrame) -> str:
    """Build a collapsible HTML data table."""
    display_cols = [c for c in [
        "trace_file", "effect_type", "timestamp", "duration_s", "iterations",
        "doc_searches", "seconds_per_iteration", "doc_searches_per_min",
    ] if c in df.columns]

    if not display_cols:
        return "<p>No data columns available for table.</p>"

    header = "".join(f"<th>{c}</th>" for c in display_cols)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in display_cols:
            val = row[c]
            if pd.isna(val):
                cells.append("<td>-</td>")
            elif isinstance(val, float):
                cells.append(f"<td>{val:.1f}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <details style="margin:20px 0;">
        <summary style="cursor:pointer;font-size:18px;font-weight:bold;color:#cdd6f4;
                        padding:10px;background:#1e1e2e;border-radius:8px;">
            Raw Data Table ({len(df)} rows)
        </summary>
        <div style="overflow-x:auto;margin-top:10px;">
            <table style="width:100%;border-collapse:collapse;font-size:13px;background:#1e1e2e;">
                <thead>
                    <tr style="background:#313244;color:#cdd6f4;">
                        {header}
                    </tr>
                </thead>
                <tbody style="color:#bac2de;">
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </details>
    """


def assemble_dashboard(
    df: pd.DataFrame,
    charts: list[tuple[str, str, go.Figure]],
    include_table: bool = True,
) -> str:
    """Assemble the full HTML dashboard."""

    # Build chart HTML sections
    chart_sections = []
    nav_links = []
    for chart_id, title, fig in charts:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#11111b",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
        )
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        chart_sections.append(f"""
        <section id="{chart_id}" style="margin:30px 0;padding-top:60px;">
            {chart_html}
        </section>
        """)
        nav_links.append(f'<a href="#{chart_id}" style="color:#89b4fa;text-decoration:none;padding:4px 12px;'
                         f'border-radius:4px;white-space:nowrap;">{title}</a>')

    nav_html = " | ".join(nav_links)
    summary_html = build_summary_stats(df)
    table_html = build_data_table(df) if include_table else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VFX Orchestrator Trace Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #11111b;
            color: #cdd6f4;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 0 20px 40px 20px;
        }}
        nav {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: #181825ee;
            backdrop-filter: blur(10px);
            padding: 12px 20px;
            border-bottom: 1px solid #313244;
            display: flex;
            align-items: center;
            gap: 8px;
            overflow-x: auto;
            font-size: 14px;
        }}
        nav a:hover {{ background: #313244; }}
        h1 {{
            text-align: center;
            padding: 30px 0 10px 0;
            font-size: 24px;
            color: #cdd6f4;
        }}
        table td, table th {{
            padding: 6px 10px;
            border-bottom: 1px solid #313244;
            text-align: right;
        }}
        table th {{ text-align: center; }}
        table td:first-child, table th:first-child {{ text-align: left; }}
        table tr:hover {{ background: #313244; }}
        details summary:hover {{ background: #313244; }}
    </style>
</head>
<body>
    <nav>
        <span style="font-weight:bold;color:#f5c2e7;margin-right:10px;">Dashboard</span>
        {nav_html}
    </nav>

    <h1>VFX Orchestrator Trace Dashboard</h1>
    {summary_html}
    {''.join(chart_sections)}
    {table_html}

    <footer style="text-align:center;color:#585b70;font-size:12px;padding:20px 0;">
        Generated by trace_visualizer.py | Data: {{len(df)}} runs
    </footer>
</body>
</html>"""


# --- Browser Open ---

def open_in_browser(filepath: str):
    """Open HTML file in browser, handling WSL2."""
    abs_path = os.path.abspath(filepath)

    # Try wslview first (WSL2)
    try:
        subprocess.run(["wslview", abs_path], check=True, timeout=5,
                       capture_output=True)
        print(f"Opened in browser via wslview: {abs_path}")
        return
    except (FileNotFoundError, PermissionError, subprocess.SubprocessError, OSError):
        pass

    # Fallback to webbrowser
    try:
        url = f"file://{abs_path}"
        webbrowser.open(url)
        print(f"Opened in browser: {url}")
        return
    except Exception:
        pass

    # Last resort: just print the path
    print(f"Could not open browser automatically.")
    print(f"Open this file manually: {abs_path}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive dashboard from trace metrics CSV"
    )
    parser.add_argument(
        "-i", "--input", default=None,
        help="Input CSV file (default: tools/metrics.csv relative to script)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output HTML file (default: {input_stem}_dashboard.html)"
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Auto-open dashboard in browser"
    )
    parser.add_argument(
        "--no-table", action="store_true",
        help="Skip data table at bottom"
    )
    args = parser.parse_args()

    # Resolve input path
    if args.input:
        csv_path = Path(args.input)
    else:
        script_dir = Path(__file__).parent
        csv_path = script_dir / "metrics.csv"

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        print("Run trace_analyzer.py --export-csv first to generate it.")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = load_and_prepare(str(csv_path))
    print(f"Loaded {len(df)} runs ({df['effect_type'].nunique()} effect types)")

    if df.empty:
        print("ERROR: No valid data rows after filtering.")
        sys.exit(1)

    # Build charts (each returns gracefully if columns are missing)
    charts = [
        ("agent-breakdown", "Agent Breakdown", build_agent_breakdown_chart(df)),
        ("duration-trends", "Duration Trends", build_duration_trends_chart(df)),
        ("doc-efficiency", "Doc Efficiency", build_doc_search_efficiency_chart(df)),
        ("iteration-cost", "Iteration Cost", build_iteration_cost_chart(df)),
        ("agent-distribution", "Agent Distribution", build_agent_distribution_chart(df)),
        ("effect-comparison", "Effect Comparison", build_effect_comparison_chart(df)),
    ]

    print("Building dashboard...")
    html = assemble_dashboard(df, charts, include_table=not args.no_table)

    # Resolve output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = csv_path.parent / f"{csv_path.stem}_dashboard.html"

    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"Dashboard saved: {out_path} ({size_kb:.0f} KB)")

    if args.open:
        open_in_browser(str(out_path))


if __name__ == "__main__":
    main()
