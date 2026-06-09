"""SAM2 interactive mask editor — Gradio web UI.

Public API:
    launch_mask_editor(frame_path, output_mask_path, device, port, open_browser)

Blocks until the user clicks **Save & Continue**, then writes a binary grayscale
PNG mask to *output_mask_path* and returns.
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from typing import Any

import numpy as np
import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level shared state
# Non-serialisable / GPU objects live here rather than in gr.State so that
# Gradio never tries to pickle or copy them between requests.
# ---------------------------------------------------------------------------
_image_np:      np.ndarray | None = None   # (H, W, 3) uint8 RGB
_predictor:     Any | None         = None   # SAM2Interactive, lazy-loaded
_object_masks:  dict[str, np.ndarray] = {}  # obj_id → (H, W) bool
_output_path:   str | None         = None
_done_event:    threading.Event | None = None
_device:        str                = "cuda"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_predictor() -> Any:
    """Return (creating if needed) the module-level SAM2Interactive instance."""
    global _predictor
    if _predictor is None:
        from tools.interact_tools import SAM2Interactive  # noqa: PLC0415
        _predictor = SAM2Interactive(device=_device)
        if _image_np is not None:
            _predictor.set_image(_image_np)
    return _predictor


def _obj_id(obj_key: str) -> str:
    """'Object 3' → '3'"""
    parts = obj_key.strip().split()
    return parts[-1] if parts else "1"


def _choices(state: dict) -> list[str]:
    return [f"Object {oid}" for oid in state["objects"]]


def _build_display(state: dict, erode_px: int, dilate_px: int) -> np.ndarray:
    """Compose annotated display image: base + all mask overlays + current-object points."""
    from tools.painter import (  # noqa: PLC0415
        apply_mask_overlay,
        draw_interaction_points,
        erode_dilate_mask,
        OBJECT_COLORS,
    )
    if _image_np is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    display = _image_np.copy()

    for obj_id, obj_data in state["objects"].items():
        mask = _object_masks.get(obj_id)
        if mask is None or not mask.any():
            continue
        cidx  = int(obj_data.get("color_idx", 0)) % len(OBJECT_COLORS)
        color = OBJECT_COLORS[cidx]
        m = erode_dilate_mask(mask.copy(), int(erode_px), int(dilate_px)) if (erode_px or dilate_px) else mask
        display = apply_mask_overlay(display, m, color_rgb=color, alpha=0.45)

    cur_id = state.get("current_id", "1")
    obj    = state["objects"].get(cur_id, {})
    pts    = obj.get("points", [])
    lbls   = obj.get("labels", [])
    if pts:
        display = draw_interaction_points(display, pts, lbls)

    return display


def _predict_for_current(state: dict) -> dict:
    """Re-run SAM2 for the active object based on its current points."""
    cur_id = state.get("current_id", "1")
    obj    = state["objects"].get(cur_id, {})
    pts    = obj.get("points", [])
    lbls   = obj.get("labels", [])

    if not pts:
        _object_masks.pop(cur_id, None)
        return state

    try:
        mask, score = _get_predictor().predict(pts, lbls, multimask=True)
        if mask is not None:
            _object_masks[cur_id] = mask
            print(f"[SAM2 Editor] Object {cur_id}: mask score={score:.3f}")
        else:
            _object_masks.pop(cur_id, None)
    except Exception as exc:
        print(f"[SAM2 Editor] Prediction error: {exc}")
        _object_masks.pop(cur_id, None)

    return state


# ---------------------------------------------------------------------------
# Gradio handler functions
# ---------------------------------------------------------------------------

def on_click(evt: gr.SelectData, state, point_mode, obj_key, erode_px, dilate_px):
    """Handle a user click on the image."""
    if evt is None or _image_np is None:
        return None, state, "No image loaded."

    idx = getattr(evt, "index", None)
    if idx is None and isinstance(evt, dict):
        idx = evt.get("index")
    if not (isinstance(idx, (list, tuple)) and len(idx) >= 2):
        return None, state, "Could not read click coordinates."

    x, y   = float(idx[0]), float(idx[1])
    label  = 1 if "Positive" in point_mode else 0
    cur_id = _obj_id(obj_key)
    state["current_id"] = cur_id

    if cur_id not in state["objects"]:
        return None, state, f"Object {cur_id} not found."

    state["objects"][cur_id]["points"].append([x, y])
    state["objects"][cur_id]["labels"].append(label)
    state = _predict_for_current(state)

    display  = _build_display(state, int(erode_px), int(dilate_px))
    n        = len(state["objects"][cur_id]["points"])
    mode_str = "✅ Positive" if label == 1 else "❌ Negative"
    status   = f"{mode_str} @ ({int(x)}, {int(y)})  •  Object {cur_id}  •  {n} point(s)"
    return display, state, status


def on_add_object(state):
    import gradio as gr
    new_id = str(state["next_id"])
    state["next_id"] += 1
    cidx = len(state["objects"])
    state["objects"][new_id] = {"points": [], "labels": [], "color_idx": cidx}
    state["current_id"] = new_id
    return state, gr.update(choices=_choices(state), value=f"Object {new_id}"), f"Added Object {new_id}"


def on_remove_object(state, obj_key):
    import gradio as gr
    rid = _obj_id(obj_key)
    if rid in state["objects"] and len(state["objects"]) > 1:
        del state["objects"][rid]
        _object_masks.pop(rid, None)
        remaining = next(iter(state["objects"]))
        state["current_id"] = remaining
    ch = _choices(state)
    val = f"Object {state['current_id']}"
    return state, gr.update(choices=ch, value=val), f"Removed Object {rid}"


def on_switch_object(obj_key, state, erode_px, dilate_px):
    cid = _obj_id(obj_key)
    state["current_id"] = cid
    return _build_display(state, int(erode_px), int(dilate_px)), state


def on_clear_points(state, obj_key, erode_px, dilate_px):
    cid = _obj_id(obj_key)
    if cid in state["objects"]:
        state["objects"][cid]["points"] = []
        state["objects"][cid]["labels"] = []
        _object_masks.pop(cid, None)
    return _build_display(state, int(erode_px), int(dilate_px)), state, f"Cleared points for Object {cid}"


def on_undo(state, obj_key, erode_px, dilate_px):
    cid = _obj_id(obj_key)
    obj = state["objects"].get(cid, {})
    if obj.get("points"):
        obj["points"].pop()
        obj["labels"].pop()
        state = _predict_for_current(state)
    return _build_display(state, int(erode_px), int(dilate_px)), state, f"Undid last point on Object {cid}"


def on_erode_dilate(state, erode_px, dilate_px):
    return _build_display(state, int(erode_px), int(dilate_px))


def on_reset_all(state, erode_px, dilate_px):
    import gradio as gr
    global _object_masks
    _object_masks.clear()
    new_state = {
        "objects":    {"1": {"points": [], "labels": [], "color_idx": 0}},
        "current_id": "1",
        "next_id":    2,
    }
    if _predictor is not None:
        try:
            _predictor.reset_image()
            if _image_np is not None:
                _predictor.set_image(_image_np)
        except Exception:
            pass
    display = _build_display(new_state, 0, 0)
    return display, new_state, gr.update(choices=["Object 1"], value="Object 1"), "Reset — all masks and points cleared."


def on_save(state, erode_px, dilate_px):
    """Combine all object masks, apply edge refinement, write PNG, unblock bridge."""
    global _done_event
    from tools.painter import erode_dilate_mask  # noqa: PLC0415

    active_masks = [m for m in _object_masks.values() if m is not None and m.any()]
    if not active_masks:
        return "⚠ No masks to save — click on the subject first."

    H, W = _image_np.shape[:2]
    combined = np.zeros((H, W), dtype=bool)
    for m in active_masks:
        combined |= m
    if int(erode_px) or int(dilate_px):
        combined = erode_dilate_mask(combined, int(erode_px), int(dilate_px))

    Image.fromarray((combined.astype(np.uint8) * 255), mode="L").save(_output_path)
    print(f"[SAM2 Editor] Mask saved → {_output_path}")

    if _done_event is not None:
        _done_event.set()

    return (
        f"✓ Mask saved!\n"
        f"Path: {_output_path}\n\n"
        "MatAnyone2 inference is now running in the terminal.\n"
        "You can close this browser tab."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def launch_mask_editor(
    frame_path:        str,
    output_mask_path:  str,
    device:            str  = "cuda",
    port:              int  = 7860,
    open_browser:      bool = True,
) -> None:
    """Launch the Gradio SAM2 editor.  **Blocks** until the user saves the mask."""
    global _image_np, _predictor, _object_masks, _output_path, _done_event, _device

    import gradio as gr

    # ── Reset module state ──────────────────────────────────────────────────
    _object_masks.clear()
    _done_event  = threading.Event()
    _output_path = output_mask_path
    _device      = device
    _predictor   = None   # force re-creation with (potentially new) device

    with Image.open(frame_path) as img:
        _image_np = np.array(img.convert("RGB"))

    H, W = _image_np.shape[:2]
    print(f"[SAM2 Editor] Frame: {frame_path}  ({W}×{H})")
    print(f"[SAM2 Editor] Output mask: {output_mask_path}")
    print("[SAM2 Editor] Loading SAM2 model — first run downloads ~2 GB checkpoint …")
    try:
        _get_predictor()
        print("[SAM2 Editor] SAM2 model loaded and image encoded.")
    except Exception as exc:
        print(f"[SAM2 Editor] WARNING — SAM2 failed to load: {exc}")
        traceback.print_exc()
        print("[SAM2 Editor] The editor will still open but segmentation will not work.")

    # ── Initial UI state ────────────────────────────────────────────────────
    init_state = {
        "objects":    {"1": {"points": [], "labels": [], "color_idx": 0}},
        "current_id": "1",
        "next_id":    2,
    }

    # ── Build Gradio layout ─────────────────────────────────────────────────
    with gr.Blocks(title="SAM2 Mask Editor") as demo:

        gr.Markdown("# 🎭 SAM2 Interactive Mask Editor")
        gr.Markdown(
            f"**Frame:** `{os.path.basename(frame_path)}` &nbsp;&nbsp;({W}×{H})  \n"
            "**Click** anywhere on the image to place SAM2 segmentation points.  "
            "Toggle **Positive / Negative** to add or subtract regions.  "
            "Use **+ Add Object** for multi-subject shots.  "
            "Click **💾 Save & Continue** when you're happy with the mask."
        )

        state      = gr.State(init_state)
        dev_state  = gr.State(device)          # read-only, passed to handlers that need it

        with gr.Row():
            # ── Left: image viewer ────────────────────────────────────────
            with gr.Column(scale=3):
                img_out = gr.Image(
                    value=_image_np.copy(),
                    label="Click to place SAM2 points",
                    interactive=True,
                    type="numpy",
                    height=min(H, 720),
                )

            # ── Right: controls panel ─────────────────────────────────────
            with gr.Column(scale=1, min_width=270):

                gr.Markdown("### 🖱 Click Mode")
                point_mode = gr.Radio(
                    choices=["✅ Positive (add to mask)", "❌ Negative (remove from mask)"],
                    value="✅ Positive (add to mask)",
                    label="",
                    container=False,
                )

                gr.Markdown("### 🗂 Objects")
                gr.Markdown(
                    "<small>Each object gets its own colour overlay.  "
                    "Combine overlapping objects freely — they merge on save.</small>"
                )
                obj_dd = gr.Dropdown(
                    choices=["Object 1"],
                    value="Object 1",
                    label="Active object",
                    interactive=True,
                )
                with gr.Row():
                    add_obj_btn = gr.Button("＋ Add Object", size="sm")
                    rem_obj_btn = gr.Button("✕ Remove",     size="sm", variant="secondary")

                gr.Markdown("### 🔧 Edge Refinement")
                erode_sl  = gr.Slider(0, 30, value=0, step=1, label="Erode edges (px)")
                dilate_sl = gr.Slider(0, 30, value=0, step=1, label="Dilate edges (px)")

                gr.Markdown("### ↩ History")
                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear Points", size="sm")
                    undo_btn  = gr.Button("↩ Undo Last",    size="sm")
                reset_btn = gr.Button("🔄 Reset All", variant="stop", size="sm")

                gr.Markdown("---")
                save_btn = gr.Button("💾 Save & Continue →", variant="primary", size="lg")
                status   = gr.Textbox(
                    label="Status",
                    value="Click on the image to start segmenting.",
                    interactive=False,
                    lines=4,
                )

        # ── Event wiring ─────────────────────────────────────────────────────
        img_out.select(
            fn=on_click,
            inputs=[state, point_mode, obj_dd, erode_sl, dilate_sl],
            outputs=[img_out, state, status],
        )
        obj_dd.change(
            fn=on_switch_object,
            inputs=[obj_dd, state, erode_sl, dilate_sl],
            outputs=[img_out, state],
        )
        add_obj_btn.click(
            fn=on_add_object,
            inputs=[state],
            outputs=[state, obj_dd, status],
        )
        rem_obj_btn.click(
            fn=on_remove_object,
            inputs=[state, obj_dd],
            outputs=[state, obj_dd, status],
        )
        erode_sl.change(
            fn=on_erode_dilate,
            inputs=[state, erode_sl, dilate_sl],
            outputs=[img_out],
        )
        dilate_sl.change(
            fn=on_erode_dilate,
            inputs=[state, erode_sl, dilate_sl],
            outputs=[img_out],
        )
        clear_btn.click(
            fn=on_clear_points,
            inputs=[state, obj_dd, erode_sl, dilate_sl],
            outputs=[img_out, state, status],
        )
        undo_btn.click(
            fn=on_undo,
            inputs=[state, obj_dd, erode_sl, dilate_sl],
            outputs=[img_out, state, status],
        )
        reset_btn.click(
            fn=on_reset_all,
            inputs=[state, erode_sl, dilate_sl],
            outputs=[img_out, state, obj_dd, status],
        )
        save_btn.click(
            fn=on_save,
            inputs=[state, erode_sl, dilate_sl],
            outputs=[status],
        )

    # ── Launch ───────────────────────────────────────────────────────────────
    print(f"[SAM2 Editor] Launching on http://127.0.0.1:{port}")
    print("[SAM2 Editor] Click on the subject, then click 'Save & Continue'.")
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=open_browser,
        prevent_thread_lock=True,
        quiet=False,
        theme=gr.themes.Soft(),
    )

    _done_event.wait()
    print("[SAM2 Editor] Mask received — resuming pipeline.")
