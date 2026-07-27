"""
Multi-user image labeling tool built with Streamlit.

- Recursively scans an image directory (folder -> subfolder -> image.jpg)
- Lets multiple reviewers (e.g. 10 people, 5 concurrently) each pull ONE
  image at a time to label
- Guarantees an image is never handed out to two people at once, using a
  SQLite table as a shared lock/assignment ledger (safe under concurrent
  access via BEGIN IMMEDIATE transactions)
- Labels are written to a CSV file: full_dir_filename,label
- Stale assignments (reviewer closed tab / went idle) are automatically
  released back into the pool after a configurable timeout
- Skip flags an image as 'skipped' so it stays out of the assignment pool
  and out of labels.csv; the sidebar has a Recycle action to bring
  skipped images back if you ever want to review them again.

Run with:
    streamlit run app.py
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from filelock import FileLock
from PIL import Image

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

IMAGE_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images_dir")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "labeling_state.db")
CSV_PATH = os.path.join(APP_DIR, "labels.csv")
CSV_LOCK_PATH = CSV_PATH + ".lock"

LABELS = ["FOP", "BOP", "OTHER_VIEW", "BOTH", "ERROR"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

STALE_TIMEOUT_MINUTES_DEFAULT = 20  # auto-release if reviewer disappears

# --------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.isolation_level = None  # manual transaction control
    return conn


@contextmanager
def immediate_transaction(conn):
    """Acquire an exclusive write lock immediately (avoids race conditions
    between concurrent Streamlit sessions/processes)."""
    conn.execute("BEGIN IMMEDIATE;")
    try:
        yield
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise


def init_db():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'unassigned',
            assigned_to TEXT,
            assigned_at TEXT,
            label TEXT,
            labeled_by TEXT,
            labeled_at TEXT
        );
        """
    )
    # Backward-compatible: add skip-tracking columns if the DB predates them.
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(images);").fetchall()
    }
    if "skipped_by" not in existing_cols:
        conn.execute("ALTER TABLE images ADD COLUMN skipped_by TEXT;")
    if "skipped_at" not in existing_cols:
        conn.execute("ALTER TABLE images ADD COLUMN skipped_at TEXT;")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON images(status);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_assigned_to ON images(assigned_to);"
    )
    conn.close()


def scan_and_populate(root_dir: str) -> int:
    """Walk root_dir (folder -> subfolder -> image) and insert any new
    image paths into the DB. Safe to call repeatedly (idempotent)."""
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTS:
                found.append(os.path.abspath(os.path.join(dirpath, fname)))

    conn = get_conn()
    with immediate_transaction(conn):
        conn.executemany(
            "INSERT OR IGNORE INTO images (path, status) VALUES (?, 'unassigned');",
            [(p,) for p in found],
        )
    conn.close()
    return len(found)


def release_stale_assignments(conn, timeout_minutes: int):
    cutoff = (datetime.utcnow() - timedelta(minutes=timeout_minutes)).isoformat()
    conn.execute(
        """
        UPDATE images
        SET status='unassigned', assigned_to=NULL, assigned_at=NULL
        WHERE status='assigned' AND assigned_at < ?;
        """,
        (cutoff,),
    )


def assign_next_image(username: str, timeout_minutes: int):
    """Atomically: release stale locks, resume the user's existing
    in-progress image if any, otherwise grab the next unassigned image and
    lock it to this user. Returns the image path or None if nothing left.

    Images with status='skipped' are excluded from the pool automatically
    (only 'unassigned' rows are considered).
    """
    conn = get_conn()
    try:
        with immediate_transaction(conn):
            release_stale_assignments(conn, timeout_minutes)

            row = conn.execute(
                "SELECT path FROM images WHERE status='assigned' AND assigned_to=? LIMIT 1;",
                (username,),
            ).fetchone()
            if row:
                return row[0]

            row = conn.execute(
                "SELECT path FROM images WHERE status='unassigned' ORDER BY id LIMIT 1;"
            ).fetchone()
            if not row:
                return None

            path = row[0]
            conn.execute(
                """
                UPDATE images
                SET status='assigned', assigned_to=?, assigned_at=?
                WHERE path=? AND status='unassigned';
                """,
                (username, datetime.utcnow().isoformat(), path),
            )
            return path
    finally:
        conn.close()


def save_label(path: str, label: str, username: str):
    conn = get_conn()
    try:
        with immediate_transaction(conn):
            conn.execute(
                """
                UPDATE images
                SET status='done', label=?, labeled_by=?, labeled_at=?
                WHERE path=? AND assigned_to=?;
                """,
                (label, username, datetime.utcnow().isoformat(), path, username),
            )
    finally:
        conn.close()
    sync_csv()


def release_current_assignment(path: str, username: str):
    """Give the image back to the pool without labeling (e.g. corrupt)."""
    conn = get_conn()
    try:
        with immediate_transaction(conn):
            conn.execute(
                """
                UPDATE images
                SET status='unassigned', assigned_to=NULL, assigned_at=NULL
                WHERE path=? AND assigned_to=?;
                """,
                (path, username),
            )
    finally:
        conn.close()


def mark_skipped(path: str, username: str):
    """Flag an image as 'skipped' so it won't be re-assigned to anyone.
    Also clears the current assignment so the user immediately moves on
    to the next image. Skipped images can later be recycled back into
    the pool via the sidebar 'recycle skipped' action.

    Skipped rows never make it into labels.csv (only status='done' rows
    are exported), so the CSV stays in sync automatically after a skip."""
    conn = get_conn()
    try:
        with immediate_transaction(conn):
            conn.execute(
                """
                UPDATE images
                SET status='skipped',
                    skipped_by=?,
                    skipped_at=?,
                    assigned_to=NULL,
                    assigned_at=NULL
                WHERE path=? AND assigned_to=?;
                """,
                (username, datetime.utcnow().isoformat(), path, username),
            )
    finally:
        conn.close()
    # Keep labels.csv authoritative: rewrite it now so any skipped row
    # that was previously 'done' (e.g. after a re-review) is removed.
    sync_csv()


def recycle_skipped():
    """Move all 'skipped' images back to 'unassigned' so they re-enter the pool."""
    conn = get_conn()
    try:
        with immediate_transaction(conn):
            cur = conn.execute(
                """
                UPDATE images
                SET status='unassigned', skipped_by=NULL, skipped_at=NULL
                WHERE status='skipped';
                """
            )
            n = cur.rowcount
    finally:
        conn.close()
    return n


def get_user_history(username: str):
    """All images this user has already labeled, oldest first (so the
    list reads chronologically — position N is the Nth image they
    labeled). The UI defaults to opening on the LAST item, i.e. the most
    recently labeled image."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT path, label, labeled_at FROM images
        WHERE status='done' AND labeled_by=?
        ORDER BY labeled_at ASC;
        """,
        (username,),
    ).fetchall()
    conn.close()
    return rows


def update_existing_label(path: str, new_label: str, username: str) -> bool:
    """Let a reviewer change a label on an image THEY already annotated.
    Only succeeds if that image is currently marked done and labeled_by
    matches this user (prevents editing someone else's annotation).
    Returns True on success. Change is immediately reflected in the CSV."""
    conn = get_conn()
    try:
        with immediate_transaction(conn):
            cur = conn.execute(
                """
                UPDATE images
                SET label=?, labeled_at=?
                WHERE path=? AND status='done' AND labeled_by=?;
                """,
                (new_label, datetime.utcnow().isoformat(), path, username),
            )
            changed = cur.rowcount > 0
    finally:
        conn.close()
    if changed:
        sync_csv()
    return changed


def skip_from_history(path: str, username: str) -> bool:
    """Flip an already-labeled image (status='done') to 'skipped'. Only
    succeeds if the current user is the one who originally labeled it,
    so nobody can skip someone else's annotation. The original label /
    labeled_by / labeled_at are preserved as historical info, and
    skipped_by / skipped_at are recorded. labels.csv is refreshed so the
    now-skipped row is removed from it."""
    conn = get_conn()
    try:
        with immediate_transaction(conn):
            cur = conn.execute(
                """
                UPDATE images
                SET status='skipped',
                    skipped_by=?,
                    skipped_at=?
                WHERE path=? AND status='done' AND labeled_by=?;
                """,
                (username, datetime.utcnow().isoformat(), path, username),
            )
            changed = cur.rowcount > 0
    finally:
        conn.close()
    if changed:
        sync_csv()
    return changed


def get_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM images;").fetchone()[0]
    done = conn.execute("SELECT COUNT(*) FROM images WHERE status='done';").fetchone()[0]
    assigned = conn.execute(
        "SELECT COUNT(*) FROM images WHERE status='assigned';"
    ).fetchone()[0]
    skipped = conn.execute(
        "SELECT COUNT(*) FROM images WHERE status='skipped';"
    ).fetchone()[0]
    per_user = conn.execute(
        """
        SELECT labeled_by, COUNT(*) FROM images
        WHERE status='done' GROUP BY labeled_by ORDER BY COUNT(*) DESC;
        """
    ).fetchall()
    conn.close()
    return {
        "total": total,
        "done": done,
        "assigned": assigned,
        "skipped": skipped,
        "remaining": total - done - assigned - skipped,
        "per_user": per_user,
    }


def sync_csv():
    """Rewrite labels.csv from the DB's 'done' rows: full_dir_filename,label.
    Uses a cross-process file lock so concurrent saves can't corrupt it.
    Skipped rows are naturally excluded because they aren't status='done'."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT path, label FROM images WHERE status='done' ORDER BY path;"
    ).fetchall()
    conn.close()

    with FileLock(CSV_LOCK_PATH, timeout=30):
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write("full_dir_filename,label\n")
            for path, label in rows:
                safe_path = path.replace('"', '""')
                f.write(f'"{safe_path}",{label}\n')


# --------------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------------

st.set_page_config(page_title="Image Labeling Tool", layout="wide")
init_db()

st.title("🖼️ Multi-Reviewer Image Labeling Tool")

# Auto-scan the hardcoded image directory once per session (idempotent —
# safe to run repeatedly; only NEW files get inserted into the DB).
if not os.path.isdir(IMAGE_ROOT_DIR):
    st.error(
        f"IMAGE_ROOT_DIR is not a valid directory: `{IMAGE_ROOT_DIR}`\n\n"
        "Edit the `IMAGE_ROOT_DIR` constant near the top of app.py to point "
        "at your images folder."
    )
    st.stop()

if "scanned" not in st.session_state:
    with st.spinner("Scanning image directory..."):
        n_found = scan_and_populate(IMAGE_ROOT_DIR)
    st.session_state["scanned"] = True
    st.session_state["scan_count"] = n_found

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Setup")

    username = st.text_input("Your name / reviewer ID", key="username")

    mode = st.radio(
        "Mode",
        ["Label new images", "Review my history"],
        key="mode",
    )

    st.caption(f"📁 Image folder: `{IMAGE_ROOT_DIR}`")
    if st.button("🔁 Rescan directory (pick up new images)"):
        with st.spinner("Rescanning..."):
            n = scan_and_populate(IMAGE_ROOT_DIR)
        st.success(f"Rescan complete. {n} images found on disk.")

    timeout_minutes = st.number_input(
        "Auto-release idle assignment after (minutes)",
        min_value=1,
        max_value=120,
        value=STALE_TIMEOUT_MINUTES_DEFAULT,
    )

    st.divider()
    st.subheader("Progress")
    stats = get_stats()
    if stats["total"] > 0:
        st.progress(stats["done"] / stats["total"])
    st.write(f"**Total:** {stats['total']}")
    st.write(f"**Done:** {stats['done']}")
    st.write(f"**In progress:** {stats['assigned']}")
    st.write(f"**Skipped:** {stats['skipped']}")
    st.write(f"**Remaining:** {stats['remaining']}")

    if stats["skipped"] > 0:
        # Recycling is a bulk destructive action, so it's gated two ways:
        #   1. Only sessions whose OS user actually has WRITE access to the
        #      DB file are allowed to trigger it (prevents the button from
        #      surfacing to reviewers running the app under a different
        #      OS user that can't write labeling_state.db anyway).
        #   2. A "CONFIRM" keyword must be typed to arm the button, so a
        #      stray click can't wipe skip decisions.
        can_write_db = os.access(DB_PATH, os.W_OK)
        st.markdown(f"**♻️ Recycle skipped ({stats['skipped']}):**")
        if not can_write_db:
            st.caption(
                "🔒 Disabled — the OS user running this app does not have "
                f"write access to `{os.path.basename(DB_PATH)}`. Ask the "
                "database owner to run this action."
            )
        else:
            confirm_text = st.text_input(
                "Type CONFIRM to enable the recycle button",
                key="recycle_confirm",
                placeholder="CONFIRM",
            )
            confirmed = confirm_text.strip() == "CONFIRM"
            if st.button(
                f"♻️ Recycle {stats['skipped']} skipped image(s) into pool",
                disabled=not confirmed,
                help=(
                    "Type CONFIRM above to enable this button."
                    if not confirmed
                    else "This will move ALL skipped images back to 'unassigned'."
                ),
            ):
                n = recycle_skipped()
                st.session_state.pop("recycle_confirm", None)
                st.success(f"{n} image(s) moved back to 'unassigned'.")
                st.rerun()

    if stats["per_user"]:
        st.subheader("Labeled by")
        for user, count in stats["per_user"]:
            st.write(f"- {user}: {count}")

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "rb") as f:
            st.download_button(
                "⬇️ Download labels.csv",
                data=f.read(),
                file_name="labels.csv",
                mime="text/csv",
            )

    if st.button("🔄 Refresh"):
        st.rerun()

# ---------------- Main area ----------------
if not username:
    st.info("Enter your name in the sidebar to start reviewing.")
    st.stop()

if stats["total"] == 0:
    st.warning("No images loaded yet. Check the image directory setting in the sidebar.")
    st.stop()

# ============================================================
# MODE: Review my history (go back and view/edit past labels)
# ============================================================
if mode == "Review my history":
    history = get_user_history(username)

    if not history:
        st.info("You haven't labeled any images yet — switch to **Label new images** to get started.")
        st.stop()

    st.subheader(f"📚 Your labeling history ({len(history)} images)")

    # Keep a stable index into the history list across reruns.
    # Default to the LAST item = the most recently labeled image, and
    # re-snap to it whenever the history grows (new labels added since
    # we last looked).
    if (
        "history_idx" not in st.session_state
        or st.session_state.get("history_len") != len(history)
    ):
        st.session_state["history_idx"] = len(history) - 1
        st.session_state["history_len"] = len(history)
    st.session_state["history_idx"] = max(
        0, min(st.session_state["history_idx"], len(history) - 1)
    )

    options = [f"{i+1}. {os.path.basename(p)}  —  {lbl}" for i, (p, lbl, _) in enumerate(history)]
    picked = st.selectbox(
        "Jump to a specific image",
        options=range(len(history)),
        format_func=lambda i: options[i],
        index=st.session_state["history_idx"],
    )
    if picked != st.session_state["history_idx"]:
        st.session_state["history_idx"] = picked
        st.rerun()

    idx = st.session_state["history_idx"]
    path, current_label, labeled_at = history[idx]

    nav_prev, nav_pos, nav_next = st.columns([1, 2, 1])
    with nav_prev:
        if st.button("⬅️ Previous", use_container_width=True, disabled=idx == 0):
            st.session_state["history_idx"] = idx - 1
            st.rerun()
    with nav_pos:
        st.markdown(
            f"<div style='text-align:center'>{idx + 1} / {len(history)}</div>",
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("Next ➡️", use_container_width=True, disabled=idx == len(history) - 1):
            st.session_state["history_idx"] = idx + 1
            st.rerun()

    col_img, col_actions = st.columns([2, 1])

    with col_img:
        try:
            img = Image.open(path)
            st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Could not open image: {e}")
        st.caption(path)

    with col_actions:
        st.write(f"**Current label:** {current_label}")
        st.caption(f"Labeled at: {labeled_at}")
        st.write("Change label:")

        for label in LABELS:
            is_current = label == current_label
            if st.button(
                f"✅ {label}" if is_current else label,
                use_container_width=True,
                key=f"hist_btn_{label}",
                disabled=is_current,
            ):
                ok = update_existing_label(path, label, username)
                if ok:
                    st.success(f"Updated to {label}.")
                    st.rerun()
                else:
                    st.error("Could not update — this label may belong to someone else.")

        st.divider()
        if st.button(
            "⏭️ Skip this image",
            use_container_width=True,
            key="hist_skip_btn",
            help=(
                "Flag this image as skipped. It will be removed from "
                "labels.csv and stay out of the assignment pool until "
                "someone recycles skipped images."
            ),
        ):
            ok = skip_from_history(path, username)
            if ok:
                st.success("Image marked as skipped.")
                # History shrank by one; keep the user near the same
                # position instead of auto-snapping to the newest item.
                new_len = len(history) - 1
                st.session_state["history_len"] = new_len
                if new_len == 0:
                    st.session_state.pop("history_idx", None)
                else:
                    st.session_state["history_idx"] = min(
                        st.session_state["history_idx"], new_len - 1
                    )
                st.rerun()
            else:
                st.error("Could not skip — this label may belong to someone else.")

    st.stop()

# ============================================================
# MODE: Label new images (default flow)
# ============================================================
current_path = assign_next_image(username, timeout_minutes)

if current_path is None:
    if stats["skipped"] > 0:
        st.info(
            f"No unassigned images left. {stats['skipped']} image(s) were skipped — "
            "use **♻️ Recycle skipped image(s) into pool** in the sidebar to review them."
        )
    else:
        st.success("🎉 All images have been labeled! Nothing left to review.")
    st.stop()

col_img, col_actions = st.columns([2, 1])

with col_img:
    try:
        img = Image.open(current_path)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Could not open image: {e}")
    st.caption(current_path)

with col_actions:
    st.subheader(f"Reviewer: {username}")
    st.write("Choose a label — this saves it and loads the next image.")

    for label in LABELS:
        if st.button(label, use_container_width=True, key=f"btn_{label}"):
            save_label(current_path, label, username)
            st.rerun()

    st.divider()
    if st.button("⏭️ Skip (flag as skipped, move on)", use_container_width=True):
        mark_skipped(current_path, username)
        st.rerun()
