"""Automate the Colab GPU training run for pi-learn-station.

Drives a headful Chromium (persistent profile under ml/.playwright-profile):

  1. First run: opens Colab and waits for a ONE-TIME manual Google login.
  2. Opens the training notebook (from the GitHub URL).
  3. Ensures a GPU runtime, connects, runs all cells.
  4. Monitors until the "===TRAINING_DONE===" sentinel appears (handling the
     free-tier "runtime disconnected" case by reconnect + rerun).
  5. Captures model.onnx / config.json / model.pt downloads into ~/Downloads,
     verifies them, and copies them into models/recog/.

Usage:
  python ml/colab_automation.py            # full run (default, headful)
  python ml/colab_automation.py --login-only   # open + wait for manual login only
  python ml/colab_automation.py --timeout 3600 # override total wait (s)

Notes:
  - Only the Google sign-in is manual (automated login is blocked by Google).
  - Colab's UI changes frequently; the script tries multiple selectors and,
    if it cannot find a control, pauses with clear on-screen instructions
    instead of guessing.
"""
import argparse
import hashlib
import os
import shutil
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE = os.path.join(os.path.dirname(__file__), ".playwright-profile")
MODELS_RECOG = os.path.join(ROOT, "models", "recog")
DOWNLOADS = os.path.expanduser("~/Downloads")
ARTIFACTS = ["model.onnx", "config.json", "model.pt"]
NOTEBOOK_URL = (
    "https://colab.research.google.com/github/"
    "sujith0613/pi-learn-station/blob/main/ml/train_colab.ipynb"
)
SENTINEL = "===TRAINING_DONE==="


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_signed_out(page):
    """True when a toolbar 'Sign in' control is present (logged out).

    Waits for the toolbar to render before deciding.
    """
    for _ in range(12):
        try:
            el = page.get_by_text("Sign in", exact=False).first
            if el.is_visible(timeout=500):
                return True
        except Exception:
            pass
        page.wait_for_timeout(500)
    return False


def is_notebook_loaded(page):
    """True when the Colab notebook is present AND the account is signed in.

    Signed-in is required to actually run cells, so a visible toolbar 'Sign in'
    control means we are NOT ready yet.
    """
    url = page.url
    if "accounts.google.com" in url or "ServiceLogin" in url:
        return False
    if "colab.research.google.com" not in url:
        return False
    if is_signed_out(page):
        return False
    # Signed in: the notebook must expose a visible runtime 'Connect' control.
    try:
        return page.get_by_role("button", name="Connect").first.is_visible(
            timeout=3000)
    except Exception:
        return False


def wait_for_login(page, timeout_sec):
    """Wait until the notebook is accessible; prompt for manual login."""
    log("Checking sign-in state...")
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if is_notebook_loaded(page):
            log("Notebook is accessible (signed in).")
            return True
        if "accounts.google.com" in page.url or is_signed_out(page):
            log("Google login required. Sign in manually in the browser "
                "window that just opened. Waiting (this persists for future "
                "runs)...")
        page.wait_for_timeout(2000)
    log("Timed out waiting for login. Re-run after signing in.")
    return False


def _select(select, label):
    try:
        select.select_option(label=label)
        return True
    except Exception:
        return False


def set_gpu_runtime(page):
    """Attempt to set hardware accelerator to GPU via the Runtime menu."""
    log("Setting runtime type to GPU...")
    try:
        # Runtime menu
        page.get_by_text("Runtime", exact=True).first.click(timeout=5000)
        page.wait_for_timeout(500)
        page.get_by_text("Change runtime type").first.click(timeout=3000)
        page.wait_for_timeout(800)
    except Exception:
        log("  Runtime menu not found; will try the connect-dialog route.")

    # The dialog: a <select> for "Hardware accelerator". Try several labels.
    tried = page.locator("select").count()
    for i in range(page.locator("select").count()):
        sel = page.locator("select").nth(i)
        try:
            opts = sel.locator("option").all_text_contents()
        except Exception:
            continue
        if any("GPU" in o for o in opts) or any("accelerator" in o.lower()
                                                for o in opts):
            log(f"  found accelerator select with options: {opts}")
            if _select(sel, "T4 GPU"):
                log("  selected T4 GPU")
            elif _select(sel, "GPU"):
                log("  selected GPU")
            else:
                # pick any option containing GPU
                for o in opts:
                    if "GPU" in o:
                        sel.select_option(label=o); log(f"  selected {o}"); break
            break

    # Save / close the dialog
    for btn in ["Save", "OK", "Done", "Save & close"]:
        try:
            page.get_by_role("button", name=btn).first.click(timeout=2000)
            log(f"  clicked '{btn}'"); break
        except Exception:
            continue
    page.wait_for_timeout(800)


def connect_and_run(page):
    """Connect the runtime and run all cells."""
    # Ensure we're connected: look for a 'Connect' button.
    try:
        page.get_by_role("button", name="Connect").first.click(timeout=4000)
        log("  clicked Connect")
        page.wait_for_timeout(6000)
    except Exception:
        log("  no Connect button (already connected or still connecting)")

    # GPU check + run all. Ctrl+F9 runs all cells.
    log("Running all cells (Ctrl+F9)...")
    try:
        page.keyboard.press("Control+F9")
        log("  sent Ctrl+F9")
    except Exception:
        log("  Ctrl+F9 failed; falling back to manual instructions.")
        manual_pause(page, "Press Ctrl+F9 (Run all) in the Colab window.")
    # Accept any confirmation dialog
    page.once("dialog", lambda d: d.accept())
    page.wait_for_timeout(1500)


def manual_pause(page, instruction):
    log("=" * 60)
    log("PAUSE — manual action required:")
    log("  " + instruction)
    log("  This script is waiting. When done, it will continue automatically.")
    log("=" * 60)
    # Just poll a few seconds so the user reads it; no hard stop.
    page.wait_for_timeout(8000)


def handle_disconnected(page):
    try:
        body = page.content()
        if "runtime disconnected" in body.lower() or "reconnect" in body.lower():
            log("Runtime disconnected detected; reconnecting...")
            for label in ["Reconnect", "Connect", "Restart runtime"]:
                try:
                    page.get_by_role("button", name=label).first.click(
                        timeout=3000)
                    log(f"  clicked '{label}'"); break
                except Exception:
                    continue
            page.wait_for_timeout(8000)
            return True
    except Exception:
        pass
    return False


def wait_for_sentinel(page, timeout_sec):
    """Poll for the sentinel; reconnect+rerun if the runtime drops."""
    t0 = time.time()
    last_stat = 0
    while time.time() - t0 < timeout_sec:
        if handle_disconnected(page):
            log("Reconnected; re-running all cells.")
            connect_and_run(page)
            continue
        try:
            body = page.content()
        except Exception:
            body = ""
        if SENTINEL in body:
            log("SENTINEL DETECTED — training complete.")
            return True
        # periodic status (a few heartbeat lines from the notebook)
        now = time.time()
        if now - last_stat > 60:
            last_stat = now
            el = int((now - t0) / 60)
            log(f"  still training... {el}m elapsed (waiting for sentinel)")
        page.wait_for_timeout(5000)
    log("Timed out waiting for training to finish.")
    return False


def collect_artifacts():
    """Move downloaded artifacts from ~/Downloads into models/recog/."""
    log("Collecting artifacts...")
    os.makedirs(MODELS_RECOG, exist_ok=True)
    moved = []
    for name in ARTIFACTS:
        src = os.path.join(DOWNLOADS, name)
        if not os.path.exists(src):
            log(f"  missing {name} in ~/Downloads"); continue
        dst = os.path.join(MODELS_RECOG, name)
        shutil.copy2(src, dst)
        moved.append((name, sha256(src)))
        log(f"  copied {name}  sha256={sha256(src)[:16]}  "
            f"{os.path.getsize(src)} bytes")
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--login-only", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()

    os.makedirs(PROFILE, exist_ok=True)
    os.makedirs(DOWNLOADS, exist_ok=True)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            PROFILE,
            headless=a.headless,
            args=["--no-sandbox"],
            accept_downloads=True,
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()

        # capture downloads
        def on_download(dl):
            name = dl.suggested_filename
            target = os.path.join(DOWNLOADS, name)
            dl.save_as(target)
            log(f"  download captured: {target} ({os.path.getsize(target)}B)")

        ctx.on("download", on_download)

        log("Opening Colab notebook...")
        page.goto(NOTEBOOK_URL, wait_until="domcontentloaded", timeout=180000)

        if not wait_for_login(page, timeout_sec=600):
            ctx.close(); sys.exit(2)

        if a.login_only:
            log("Login confirmed. Notebook is ready. Exiting (--login-only).")
            ctx.close(); return

        set_gpu_runtime(page)
        connect_and_run(page)

        ok = wait_for_sentinel(page, timeout_sec=a.timeout)
        ctx.close()

        if ok:
            moved = collect_artifacts()
            log("Done. Artifacts moved to models/recog/:")
            for name, h in moved:
                log(f"  {name}  sha256={h}")
            sys.exit(0 if len(moved) == len(ARTIFACTS) else 3)
        else:
            log("Training did not finish in the allotted time.")
            sys.exit(1)


if __name__ == "__main__":
    main()