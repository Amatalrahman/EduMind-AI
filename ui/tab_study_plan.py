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


def select_for_review(topic: str):
    """Callback to pre-fill the topic across all study modes."""
    st.session_state["quiz_topic"] = topic
    st.session_state["fc_topic"] = topic
    st.session_state["rt_topic_input"] = topic
    st.toast(f"Topic '{topic}' loaded! Switch to Quizzes, Flashcards, or Explain tabs to start.", icon="🚀")


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

    # Tabs inside the Study Plan
    tab_cal, tab_list, tab_stats = st.tabs(["📆 7-Day Calendar", "📋 Full List", "📊 Performance"])

    now = datetime.now()
    today = now.date()

    # Pre-calculate review dates
    entries_with_dates = []
    for entry in log:
        next_review_dt = _sm2_next_review(
            times_reviewed=entry["times_reviewed"],
            quiz_accuracy=entry["quiz_accuracy"],
            last_seen_str=entry["last_seen"],
        )
        # If overdue, group it onto Today
        review_date = next_review_dt.date()
        if review_date < today:
            review_date = today
        entries_with_dates.append({**entry, "next_review_dt": next_review_dt, "review_date": review_date})

    # ── 1. Calendar View ────────────────────────────────────────────────────────
    with tab_cal:
        st.subheader("🗓️ Up Next (7 Days)")
        
        # Group by date
        from collections import defaultdict
        scheduled = defaultdict(list)
        for e in entries_with_dates:
            scheduled[e["review_date"]].append(e)

        days = [today + timedelta(days=i) for i in range(7)]
        cols = st.columns(7)

        for i, day in enumerate(days):
            with cols[i]:
                # Day header
                if i == 0:
                    st.markdown("**Today**")
                elif i == 1:
                    st.markdown("**Tomorrow**")
                else:
                    st.markdown(f"**{day.strftime('%a')}**<br>{day.strftime('%m/%d')}", unsafe_allow_html=True)
                
                st.divider()
                
                topics = scheduled.get(day, [])
                if not topics:
                    st.caption("No reviews")
                else:
                    for t in topics:
                        with st.container(border=True):
                            st.write(f"**{t['topic']}**")
                            # Color dot based on accuracy
                            acc = t['quiz_accuracy']
                            dot = "🔴" if acc < 0.5 else "🟡" if acc < 0.8 else "🟢"
                            st.caption(f"{dot} {acc*100:.0f}% acc")
                            st.button(
                                "Study",
                                key=f"study_{t['id']}_{day}",
                                on_click=select_for_review,
                                args=(t['topic'],),
                                use_container_width=True,
                            )

    # ── 2. Full List View ───────────────────────────────────────────────────────
    with tab_list:
        st.subheader("Review Schedule (All)")
        for entry in entries_with_dates:
            next_review_dt = entry["next_review_dt"]
            days_over = _days_overdue(next_review_dt)
            accuracy_pct = entry["quiz_accuracy"] * 100

            if days_over > 0:
                urgency = "🔴"
                label = f"**OVERDUE by {days_over} day(s)**"
            elif (next_review_dt - now).days <= 1:
                urgency = "🟡"
                label = "**Due today or tomorrow**"
            else:
                urgency = "🟢"
                label = f"Next review: {next_review_dt.strftime('%Y-%m-%d')}"

            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
                col1.markdown(f"{urgency} **{entry['topic']}**")
                col2.metric("Quiz accuracy", f"{accuracy_pct:.0f}%")
                col3.metric("Reviews done", entry["times_reviewed"])
                with col4:
                    st.button(
                        "Study Now",
                        key=f"list_study_{entry['id']}",
                        on_click=select_for_review,
                        args=(entry['topic'],),
                        use_container_width=True,
                    )
                st.caption(
                    f"{label} | Last reviewed: {entry['last_seen'][:10]}"
                )

    # ── 3. Performance Stats ────────────────────────────────────────────────────
    with tab_stats:
        st.subheader("📊 Overview")
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
