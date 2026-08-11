"""
Study Plan tab — SM-2 spaced repetition scheduling from long-term memory.
Surfaces topics with lowest quiz accuracy and longest time since last review.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

from db.database import get_all_subjects, get_study_log_by_subject


def _sm2_next_review(times_reviewed: int, quiz_accuracy: float, last_seen_str: str) -> datetime:
    """
    Simplified SM-2: interval grows with accuracy, shrinks with poor performance.
    Returns the recommended next review datetime.
    """
    try:
        last_seen = datetime.fromisoformat(last_seen_str)
    except Exception:
        last_seen = datetime.now()

    # Ease factor (SM-2 style): higher accuracy → longer interval
    ef = max(1.3, 2.5 + 0.1 - 0.08 * (1.0 - quiz_accuracy) * 5)

    if times_reviewed == 0:
        interval_days = 1
    elif times_reviewed == 1:
        interval_days = 6
    else:
        # Approximate: each subsequent review multiplies by EF
        interval_days = round(6 * (ef ** (times_reviewed - 1)))

    # Cap at 365 days
    interval_days = min(interval_days, 365)
    return last_seen + timedelta(days=interval_days)


def _days_overdue(next_review: datetime) -> int:
    delta = (datetime.now() - next_review).days
    return max(0, delta)


def render_study_plan_tab():
    st.header("📅 Study Plan")
    st.caption(
        "Your personalized spaced-repetition schedule based on quiz performance. "
        "Topics with lowest accuracy appear first."
    )

    subjects = get_all_subjects()
    if not subjects:
        st.info("No subjects yet. Upload documents and take quizzes to build your study plan.")
        return

    subject_names = [s["name"] for s in subjects]
    selected_name = st.selectbox("Subject", options=subject_names, key="sp_subject")
    selected_subject = next(s for s in subjects if s["name"] == selected_name)
    subject_id = selected_subject["id"]

    log = get_study_log_by_subject(subject_id)
    if not log:
        st.info(
            "No study history yet for this subject. "
            "Complete quizzes and flashcard sessions to populate your plan."
        )
        return

    st.subheader(f"Review Schedule — {selected_name}")

    now = datetime.now()
    for entry in log:
        next_review = _sm2_next_review(
            times_reviewed=entry["times_reviewed"],
            quiz_accuracy=entry["quiz_accuracy"],
            last_seen_str=entry["last_seen"],
        )
        days_over = _days_overdue(next_review)
        accuracy_pct = entry["quiz_accuracy"] * 100

        if days_over > 0:
            urgency = "🔴"
            label = f"**OVERDUE by {days_over} day(s)**"
        elif (next_review - now).days <= 1:
            urgency = "🟡"
            label = "**Due today or tomorrow**"
        else:
            urgency = "🟢"
            label = f"Next review: {next_review.strftime('%Y-%m-%d')}"

        with st.container(border=True):
            col1, col2, col3 = st.columns([4, 2, 2])
            col1.markdown(f"{urgency} **{entry['topic']}**")
            col2.metric("Quiz accuracy", f"{accuracy_pct:.0f}%")
            col3.metric("Reviews done", entry["times_reviewed"])
            st.caption(
                f"{label} | Last reviewed: {entry['last_seen'][:10]} | "
                f"Reviewed {entry['times_reviewed']}x"
            )

    st.divider()
    st.subheader("📊 Performance Overview")
    if log:
        import pandas as pd
        df = pd.DataFrame(log)
        df["quiz_accuracy_pct"] = (df["quiz_accuracy"] * 100).round(1)
        df = df[["topic", "quiz_accuracy_pct", "times_reviewed", "last_seen"]].rename(
            columns={
                "topic": "Topic",
                "quiz_accuracy_pct": "Accuracy (%)",
                "times_reviewed": "Reviews",
                "last_seen": "Last Seen",
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
