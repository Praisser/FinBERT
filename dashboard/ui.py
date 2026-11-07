"""HTML building blocks for the dashboard. Their look is defined in style.css."""

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

CSS = Path(__file__).with_name("style.css").read_text()


def inject_css() -> None:
    st.html(f"<style>{CSS}</style>")


def day(value) -> str:
    date = pd.Timestamp(value)
    return f"{date.day} {date:%b %Y}"


def short_day(value) -> str:
    date = pd.Timestamp(value)
    return f"{date.day} {date:%b}"


def pct(change: float) -> str:
    return f"{change:+.2%}"


def panel(name: str):
    """A container drawn with a hairline border (see [class*="st-key-panel_"] in style.css)."""
    return st.container(key=f"panel_{name}")


def section(title: str, text: str = "") -> None:
    paragraph = f"<p>{text}</p>" if text else ""
    st.html(f'<div class="fx-section"><div class="fx-title">{title}</div>{paragraph}</div>')


def panel_header(title: str, note: str = "") -> None:
    note_html = f'<div class="fx-panel-note">{note}</div>' if note else ""
    st.html(f'<div class="fx-panel-title">{title}</div>{note_html}')


def figures(items: list[tuple[str, str, str]]) -> str:
    """A row of (label, value, note) figures separated by hairlines."""
    cells = "".join(
        f'<div class="fx-figure"><div class="fx-label">{label}</div>'
        f'<div class="fx-figure-value">{value}</div><div class="fx-figure-note">{note}</div></div>'
        for label, value, note in items
    )
    return f'<div class="fx-figures" style="--n:{len(items)}">{cells}</div>'


def headline_list(news: pd.DataFrame, limit: int = 40) -> str:
    latest = news.sort_values("published", ascending=False).head(limit)
    rows = "".join(
        f'<div class="fx-row"><div class="fx-row-time">{short_day(published)}, {published:%H:%M}</div>'
        f'<div class="fx-row-text">{escape(headline)}</div><div class="fx-row-tag {label}">{label.capitalize()}</div></div>'
        for published, headline, label in zip(latest["published"], latest["headline"], latest["label"])
    )
    return f'<div class="fx-list">{rows}</div>'


def table(headers: list[tuple[str, str]], rows: list[tuple[list[str], str]]) -> str:
    """HTML table. headers: (title, css class); rows: (cells, row css class)."""
    head = "".join(f'<th class="{cls}">{title}</th>' for title, cls in headers)
    body = "".join(
        f'<tr class="{row_cls}">' + "".join(f'<td class="{cls}">{cell}</td>' for cell, (_, cls) in zip(cells, headers)) + "</tr>"
        for cells, row_cls in rows
    )
    return f'<div class="fx-table-wrap"><table class="fx-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def side_label(text: str) -> None:
    st.html(f'<div class="fx-label fx-side-label">{text}</div>')
