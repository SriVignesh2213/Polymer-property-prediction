"""Reusable Plotly visualization helpers."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def bar_chart(values, title, x=None, y=None):
    fig = go.Figure(go.Bar(x=x if x is not None else list(range(len(values))), y=values))
    fig.update_layout(title=title, template="plotly_dark", height=350)
    return fig


def histogram(df, column, title):
    fig = px.histogram(df, x=column, nbins=40, title=title, template="plotly_dark")
    return fig