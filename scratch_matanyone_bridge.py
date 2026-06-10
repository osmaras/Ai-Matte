# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.31.0",
#     "imageio",
#     "pillow>=10.2.0",
#     "tqdm>=4.66.0",
#     "torch>=2.2.0",
#     "torchvision>=0.17.0",
#     "gradio>=4.26.0",
#     "opencv-python-headless>=4.8.0",
#     "sam-2 @ git+https://github.com/facebookresearch/sam2.git",
#     "matanyone2 @ git+https://github.com/osmaras/MatAnyone2.git",
#     "assimilate_client @ git+https://github.com/Assimilate-Inc/Assimilate-REST.git",
# ]
# ///

import argparse
import contextlib
import os
import sys
import time
import torch
import shutil
import requests
from PIL import Image
from tqdm.auto import tqdm

# Official Assimilate V2 API bindings
from assimilate_client import Configuration, ApiClient, DeleteMediaData, ShotData
from assimilate_client.api import SystemApi, ProjectsApi, ApplicationApi
from assimilate_client.rest import ApiException
from matanyone2 import MatAnyone2, InferenceCore

# Configuration Constants
SCRATCH_HOST = os.environ.get("SCRATCH_HOST", "http://127.0.0.1:8080")
BASE_CACHE_DIR = os.environ.get("MATANYONE_CACHE", "C:/MatAnyone_Scratch_Cache")


def _print_step(message):
    print(f"[MatAnyone Bridge] {message}")


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run the MatAnyone bridge against the active or specified SCRATCH shot."
    )
    parser.add_argument("-project", help="SCRATCH project identifier", default=None)
    parser.add_argument("-group", help="SCRATCH group UUID or name", default=None)
    parser.add_argument("-construct", help="SCRATCH construct UUID", default=None)
    parser.add_argument("-shot", help="SCRATCH shot UUID", default=None)
    parser.add_argument(
        "--scratch-host",
        default=SCRATCH_HOST,
        help=f"SCRATCH REST host base URL (default: {SCRATCH_HOST})",
    )
    parser.add_argument(
        "--cache-dir",
        default=BASE_CACHE_DIR,
        help=f"Local cache directory for rendered frames (default: {BASE_CACHE_DIR})",
    )
    parser.add_argument(
        "--max-min-side",
        type=int,
        default=1080,
        help="Resize only if min(width,height) exceeds this value. Aspect ratio is preserved.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=4,
        help="Frame overlap between VRAM batches.",
    )
    parser.add_argument(
        "--import-mode",
        choices=["api-only", "api-fallback", "manual"],
        default="api-fallback",
        help="How to load generated matte back into SCRATCH.",
    )
    parser.add_argument(
        "--note-status",
        type=int,
        default=2,
        help="Status integer used when writing the shot note.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary render and batch folders for debugging.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA GPU inference is not available.",
    )
    parser.add_argument(
        "--skip-sam",
        action="store_true",
        help="Skip the SAM2 interactive editor and reuse an existing mask.png if present.",
    )
    parser.add_argument(
        "--sam-port",
        type=int,
        default=7860,
        help="Local port for the Gradio SAM2 editor (default: 7860).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically when launching the SAM2 editor.",
    )
    return parser


def _safe_attr(value, attr_name, default=None):
    return getattr(value, attr_name, default) if value is not None else default


def _extract_selected_entries(selected_data):
    if selected_data is None:
        return []

    for attr in ("selection", "selected_shots", "shots"):
        value = _safe_attr(selected_data, attr, None)
        if isinstance(value, list) and value:
            return value
    return []


def resolve_target_shot(projects_api, app_api, args):
    if args.project:
        _print_step(f"Entering project context: {args.project}")
        app_api.do_application_project_enter(args.project)

    if args.construct:
        _print_step(f"Entering construct context: {args.construct}")
        app_api.do_application_player_enter_timeline(args.construct)

    if args.group:
        _print_step(f"Group argument received but not applied by bridge API: {args.group}")

    if args.shot:
        shot = projects_api.get_shot(shot_uuid=args.shot)
        _print_step(f"Targeted explicit shot UUID: {shot.uuid}")
        return shot

    try:
        selected = projects_api.get_construct_current_selected_shots(level="ALL")
        selections = _extract_selected_entries(selected)
        if selections:
            first = selections[0]
            selected_shot = _safe_attr(first, "shot", None)
            selected_uuid = _safe_attr(selected_shot, "uuid", None) or _safe_attr(first, "uuid", None)
            if selected_uuid:
                shot = projects_api.get_shot(shot_uuid=selected_uuid)
                _print_step(f"Targeted selected shot UUID from construct selection: {shot.uuid}")
                return shot
    except Exception as error:
        _print_step(f"Selection query unavailable, falling back to slot/version: {error}")

    shot = projects_api.get_construct_current_slot_version(slot_idx=0, version_idx=0)
    _print_step(f"Targeted fallback shot UUID from slot/version 0/0: {shot.uuid}")
    return shot


def parse_version_tuple(version_text):
    parts = []
    for token in str(version_text).split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts[:3] + [0] * max(0, 3 - len(parts)))


def log_version_support(server_version):
    current = parse_version_tuple(server_version)
    documented = (1, 0, 5)
    if current < documented:
        _print_step(
            "Warning: server REST version is older than currently published client docs "
            f"({server_version} < 1.0.5). Documented output APIs will be tried first, "
            "but media import and output field behavior may differ on this build."
        )


def extract_output_nodes(outputs_data):
    if not outputs_data:
        return []

    for attr in ("outputs", "shots"):
        value = getattr(outputs_data, attr, None)
        if value:
            return list(value)

    if isinstance(outputs_data, list):
        return outputs_data

    return []


def ensure_current_output(projects_api):
    outputs_data = projects_api.get_construct_current_outputs(level="ALL")
    outputs = extract_output_nodes(outputs_data)
    if outputs:
        target_output = outputs[0]
        _print_step(f"Targeted output node [{target_output.name}] UUID: {target_output.uuid}")
        return target_output.uuid

    _print_step("No output node found on active construct, creating one")
    created_output = projects_api.add_construct_current_output(ShotData(), level="ALL")
    output_uuid = getattr(created_output, "uuid", None)
    if not output_uuid:
        raise RuntimeError("SCRATCH created an output node but did not return its UUID.")

    output_name = getattr(created_output, "name", "MatAnyone_Output")
    _print_step(f"Created output node [{output_name}] UUID: {output_uuid}")
    return output_uuid


def create_matanyone_output(config, projects_api, shot_uuid, shot_length, shot_name):
    """Create or reuse a dedicated output node for MatAnyone matte extraction.

    If a MatAnyone output node already exists from a previous run, reuses it.
    Otherwise creates a new one via POST /constructs/{construct_uuid}/outputs/new.
    Output goes into the project's media folder with RGBA TIFF format.

    Returns (output_uuid, output_path).
    """
    # Get project media path
    proj = projects_api.get_projects_current()
    media_path = None
    try:
        pp = proj.project_paths if hasattr(proj, "project_paths") else (proj.get("project_paths") if isinstance(proj, dict) else {})
        if hasattr(pp, "media_path"):
            media_path = pp.media_path
        elif isinstance(pp, dict):
            media_path = pp.get("media_path")
    except Exception:
        pass
    if not media_path:
        raise RuntimeError("Could not determine SCRATCH project media path. "
                           "Ensure a project is loaded and has a media path configured.")

    shot_label = shot_name or shot_uuid[:8]
    output_dir = os.path.join(media_path, "MatAnyone", shot_uuid[:8], "source_frames").replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing MatAnyone output node and reuse it
    try:
        existing = extract_output_nodes(projects_api.get_construct_current_outputs(level="ALL"))
        for node in existing:
            node_name = str(getattr(node, "name", "") or "")
            node_uuid = str(getattr(node, "uuid", "") or "")
            if node_name.startswith("MatAnyone_") and node_uuid:
                _print_step(f"Reusing existing output node [{node_name}] UUID: {node_uuid}")
                return node_uuid, output_dir
    except Exception as e:
        _print_step(f"Output scan failed (non-fatal): {e}")

    # No existing node found — create via CC endpoint with type_uuid
    _print_step(f"Creating output node for shot {shot_label}")
    endpoint = f"{config.host}/constructs/current/outputs/new"
    body = {
        "name": f"MatAnyone_{shot_label}",
        "type_uuid": "00000000-0000-0000-0000-000000000004",
    }
    resp = requests.post(endpoint, json=body, timeout=20)
    resp.raise_for_status()
    result = resp.json()
    out_uuid = result.get("uuid") if isinstance(result, dict) else getattr(result, "uuid", None)
    if not out_uuid:
        raise RuntimeError("SCRATCH created the output node but did not return a UUID.")
    _print_step(f"Created output node [MatAnyone_{shot_label}] UUID: {out_uuid}")
    _print_step(f"  Output path: {output_dir}")
    return out_uuid, output_dir


def configure_current_output(config, output_uuid, output_path, single_frame=False):
    endpoint = f"{config.host}/constructs/current/outputs/{output_uuid}"
    response = requests.get(endpoint)
    response.raise_for_status()
    payload = response.json()

    output_block = payload.setdefault("output", {})
    output_block["outputpath"] = output_path

    if single_frame:
        payload["length"] = 1
        handles = payload.setdefault("handles", {})
        handles["frame_in"] = 0
        handles["frame_out"] = 0
        handles["start"] = handles.get("start", "")
        handles["end"] = handles.get("end", "")

    update_response = requests.put(endpoint, json=payload)
    update_response.raise_for_status()
    return update_response.json()


def _extract_render_queue_items(queue_data):
    if queue_data is None:
        return []
    if isinstance(queue_data, list):
        return queue_data
    for attr in ("items", "render_queue", "queue"):
        value = getattr(queue_data, attr, None)
        if isinstance(value, list):
            return value
    return []


def _http_get_json(config, path, timeout_seconds=5):
    url = f"{config.host}{path}"
    try:
        response = requests.get(url, timeout=timeout_seconds)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def _http_post_json(config, path, payload=None, timeout_seconds=5):
    url = f"{config.host}{path}"
    try:
        response = requests.post(url, json=payload or {}, timeout=timeout_seconds)
        if response.status_code in (404, 405):
            return None
        response.raise_for_status()
        return response.json() if response.content else {}
    except requests.RequestException:
        return None


def _http_post_attempt(config, path, payload=None, timeout_seconds=20):
    url = f"{config.host}{path}"
    try:
        if payload is None:
            response = requests.post(url, timeout=timeout_seconds)
        else:
            response = requests.post(url, json=payload, timeout=timeout_seconds)
        detail = (response.text or "").strip().replace("\n", " ")
        return response.status_code, detail[:240]
    except requests.RequestException as error:
        return None, str(error)


def _try_start_render_via_rest(config, output_uuid, delete_existing_media=True):
    payload = {"delete_existing_media": bool(delete_existing_media)}
    attempts = [
        ("add+start item, no payload", f"/application/render/{output_uuid}", None),
        ("add+start item, with payload", f"/application/render/{output_uuid}", payload),
        ("item start, no payload", f"/application/render/start/{output_uuid}", None),
        ("item start, with payload", f"/application/render/start/{output_uuid}", payload),
        ("global start, no payload", "/application/render/start", None),
        ("global start, with payload", "/application/render/start", payload),
    ]

    for label, path, body in attempts:
        status, detail = _http_post_attempt(config, path, payload=body, timeout_seconds=20)
        _print_step(f"Start attempt [{label}] => status={status}, detail={detail}")
        if status is not None and 200 <= int(status) < 300:
            return label

    raise RuntimeError("All documented render-start endpoint attempts failed")


def _render_item_status(item_data):
    if item_data is None:
        return ""
    if isinstance(item_data, dict):
        return str(item_data.get("status", "")).lower()
    return str(getattr(item_data, "status", "")).lower()


def _render_item_path(item_data):
    if item_data is None:
        return None
    if isinstance(item_data, dict):
        return item_data.get("path")
    return getattr(item_data, "path", None)


def _collect_file_signatures(root_dirs, extensions):
    signatures = {}
    for root_dir in root_dirs:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for walk_root, _dirs, files in os.walk(root_dir):
            for file_name in files:
                if not file_name.lower().endswith(extensions):
                    continue
                file_path = os.path.join(walk_root, file_name)
                try:
                    stat = os.stat(file_path)
                except OSError:
                    continue
                signatures[file_path] = (stat.st_mtime_ns, stat.st_size)
    return signatures


def _count_changed_files(baseline_signatures, current_signatures):
    changed = 0
    for path, current_sig in current_signatures.items():
        base_sig = baseline_signatures.get(path)
        if base_sig != current_sig:
            changed += 1
    return changed


def _infer_render_probe_dirs(config, output_uuid):
    probe_dirs = []

    queue_item = _http_get_json(config, f"/application/render/{output_uuid}", timeout_seconds=5)
    queue_path = _render_item_path(queue_item)
    if queue_path:
        probe_dirs.append(queue_path)

    output_item = _http_get_json(config, f"/constructs/current/outputs/{output_uuid}", timeout_seconds=5)
    if isinstance(output_item, dict):
        output_block = output_item.get("output") or {}
        output_path = output_block.get("outputpath")
        if output_path:
            probe_dirs.append(output_path)

            output_name = output_item.get("name")
            if output_name:
                probe_dirs.append(os.path.join(output_path, output_name))

        file_path = output_item.get("file")
        if file_path:
            probe_dirs.append(os.path.dirname(file_path))

    deduped = []
    seen = set()
    for path in probe_dirs:
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            deduped.append(path)
    return deduped


def wait_for_render_files(
    probe_dirs,
    render_started_at,
    expected_min_files,
    timeout_seconds=1800,
    check_interval=3.0,
    baseline_signatures=None,
    stalled_retry_callback=None,
    retry_after_seconds=30.0,
):
    """Wait for render completion by watching the filesystem only.
    Makes zero REST/API calls so SCRATCH's render thread is never blocked.
    """
    probe_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".dpx", ".exr")
    start_time = time.time()
    prev_count = -1
    prev_changed = -1
    stable_streak = 0
    retried_once = False
    expected_total = max(1, int(expected_min_files))
    progress = tqdm(total=expected_total, desc="Render", unit="file", leave=True)

    if baseline_signatures is None:
        baseline_signatures = {}

    try:
        while True:
            current_signatures = _collect_file_signatures(probe_dirs, probe_extensions)
            changed_count = _count_changed_files(baseline_signatures, current_signatures)

            files = recent_render_files(probe_dirs, render_started_at, probe_extensions)
            if not files and changed_count > 0:
                # Some SCRATCH render paths can rewrite files without mtime patterns
                # that pass our started_at filter. Treat changed files as progress.
                files = sorted(list(current_signatures.keys()))
            count = len(files)

            effective_progress = min(expected_total, max(count, changed_count))
            progress.n = effective_progress
            progress.set_postfix(found=count, changed=changed_count)
            progress.refresh()

            if count != prev_count:
                _print_step(f"Render file progress: {count}/{expected_min_files}")
                prev_count = count

            if changed_count != prev_changed:
                _print_step(f"Render changed files detected: {changed_count}")
                prev_changed = changed_count

            if max(count, changed_count) >= expected_total:
                stable_streak += 1
                if stable_streak >= 2:
                    _print_step(f"Render complete: {count} file(s) found")
                    return files
            else:
                stable_streak = 0

            elapsed = time.time() - start_time
            if (
                not retried_once
                and stalled_retry_callback is not None
                and elapsed >= float(retry_after_seconds)
                and max(count, changed_count) == 0
            ):
                retried_once = True
                _print_step("No render file activity detected; retrying render start (fallback)")
                try:
                    stalled_retry_callback()
                except Exception as error:
                    _print_step(f"Fallback render start failed: {error}")

            if elapsed > timeout_seconds:
                if count > 0:
                    _print_step(f"Render timeout but {count} file(s) found, continuing")
                    return files
                raise RuntimeError(f"No rendered files after {int(elapsed)}s in: {probe_dirs}")

            time.sleep(check_interval)
    finally:
        progress.close()


def recent_render_files(root_dirs, started_at, extensions):
    recent_files = set()
    for root_dir in root_dirs:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for walk_root, _dirs, files in os.walk(root_dir):
            for file_name in files:
                file_path = os.path.join(walk_root, file_name)
                if not file_name.lower().endswith(extensions):
                    continue
                if os.path.getmtime(file_path) + 1.0 >= started_at:
                    recent_files.add(os.path.normpath(file_path))
    return sorted(recent_files)


def _api_patch_json(config, path, payload):
    url = f"{config.host}{path}"
    response = requests.patch(url, json=payload, timeout=20)
    response.raise_for_status()
    return response


def _api_put_json(config, path, payload):
    url = f"{config.host}{path}"
    response = requests.put(url, json=payload, timeout=20)
    response.raise_for_status()
    return response


def _api_post_json(config, path, payload):
    url = f"{config.host}{path}"
    response = requests.post(url, json=payload, timeout=20)
    response.raise_for_status()
    return response


def _get_shot_layers(projects_api, shot_uuid):
    layers_data = projects_api.get_shot_layers(shot_uuid=shot_uuid)
    return list(getattr(layers_data, "layers", []) or [])


def _pick_mask_layer(layers, mask_layer_name, fallback_mode):
    preferred_names = [name.strip().lower() for name in str(mask_layer_name).split(",") if name.strip()]
    for idx, layer in enumerate(layers):
        current_name = str(getattr(layer, "name", "")).strip().lower()
        if current_name in preferred_names:
            return idx

    if fallback_mode == "strict":
        return None

    if fallback_mode == "first" and layers:
        return 0

    if fallback_mode == "top-visible":
        for idx in range(len(layers) - 1, -1, -1):
            if bool(getattr(layers[idx], "active", False)):
                return idx

    if fallback_mode == "top" and layers:
        return len(layers) - 1

    return None


def _set_layer_active(config, shot_uuid, layer_idx, active):
    layer_path = f"/shot/{shot_uuid}/layers/{layer_idx}"
    payload = {"active": bool(active)}

    try:
        _api_patch_json(config, layer_path, payload)
        return
    except requests.HTTPError as error:
        if error.response is None or error.response.status_code not in (404, 405):
            raise

    # Compatibility fallback for older/newer server behavior where only PUT is accepted.
    _api_put_json(config, layer_path, payload)


def _render_output_pass(app_api, output_uuid, expected_min_files=1):
    config = app_api.api_client.configuration

    # Resolve probe dirs BEFORE firing the render.
    # SCRATCH 1.0.3: any REST call made while a render is processing blocks the
    # render thread. We resolve everything we need upfront, then make NO API calls
    # while waiting — we watch only the filesystem.
    probe_dirs = _infer_render_probe_dirs(config, output_uuid)
    if probe_dirs:
        _print_step(f"Render output directories: {probe_dirs}")

    probe_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".dpx", ".exr")
    baseline_signatures = _collect_file_signatures(probe_dirs, probe_extensions)

    delete_payload = DeleteMediaData()
    delete_payload.delete_existing_media = True
    render_started_at = time.time()

    # Queue hygiene for older servers: stale queue entries can block starts.
    # Best-effort only; errors are intentionally ignored.
    try:
        app_api.do_application_render_stop()
    except Exception:
        pass
    try:
        app_api.delete_application_render_queue_item(output_uuid=output_uuid)
    except Exception:
        pass
    try:
        app_api.do_application_render_delete_media_item(output_uuid=output_uuid)
        _print_step("Cleared existing media for output")
    except Exception:
        pass

    selected_start = _try_start_render_via_rest(config, output_uuid, delete_existing_media=True)
    _print_step(f"Render fired ({selected_start})")

    def _fallback_restart_render():
        try:
            app_api.delete_application_render_queue_item(output_uuid=output_uuid)
        except Exception:
            pass

        selected = _try_start_render_via_rest(config, output_uuid, delete_existing_media=True)
        _print_step(f"Fallback fired ({selected})")

    # From this point: ZERO REST calls until render is done.
    # Watch filesystem only so SCRATCH render thread is never interrupted.
    # Exception: one fallback global start if no file activity appears.
    wait_for_render_files(
        probe_dirs,
        render_started_at,
        expected_min_files=max(1, int(expected_min_files)),
        baseline_signatures=baseline_signatures,
        stalled_retry_callback=_fallback_restart_render,
    )

    path = probe_dirs[0] if probe_dirs else None
    return {"status": "finished", "path": path, "uuid": str(output_uuid)}, render_started_at


def _cleanup_temp_workspace(workspace_dir, keep_paths):
    if not os.path.isdir(workspace_dir):
        return

    workspace_abs = os.path.abspath(workspace_dir)
    keep_abs = set()
    for path in keep_paths:
        current = os.path.abspath(path)
        # Keep the requested path and its parents up to workspace root.
        while True:
            keep_abs.add(current)
            if current == workspace_abs:
                break
            parent = os.path.abspath(os.path.dirname(current))
            if parent == current:
                break
            current = parent

    for entry in os.listdir(workspace_dir):
        full_path = os.path.abspath(os.path.join(workspace_dir, entry))
        if full_path in keep_abs:
            continue
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
        except Exception:
            pass


def _to_binary_mask(source_path, target_path):
    with Image.open(source_path) as image:
        gray = image.convert("L")
        binary = gray.point(lambda px: 255 if px >= 128 else 0, mode="L")
        binary.save(target_path)


def _resize_preserving_aspect(image, target_min_side):
    width, height = image.size
    current_min = min(width, height)
    if target_min_side <= 0 or current_min <= target_min_side:
        return image, (width, height)

    scale = float(target_min_side) / float(current_min)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return image.resize((new_w, new_h), Image.BILINEAR), (width, height)


def _prepare_batch_input(batch_frames, batch_input_dir, target_min_side):
    original_sizes = {}
    target_size = None  # computed once from first frame, applied to all
    for frame_path in batch_frames:
        base_name = os.path.basename(frame_path)
        destination = os.path.join(batch_input_dir, base_name)
        with Image.open(frame_path) as src:
            rgb = src.convert("RGB")
            original_sizes[base_name] = rgb.size
            if target_size is None:
                # Compute target size from first frame
                w, h = rgb.size
                current_min = min(w, h)
                if target_min_side > 0 and current_min > target_min_side:
                    scale = float(target_min_side) / float(current_min)
                    target_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                else:
                    target_size = (w, h)
            if rgb.size != target_size:
                rgb = rgb.resize(target_size, Image.BILINEAR)
            rgb.save(destination)
    return original_sizes


def _restore_alpha_size(alpha_src_path, alpha_dst_path, expected_size):
    with Image.open(alpha_src_path) as alpha_image:
        gray = alpha_image.convert("L")
        if gray.size != expected_size:
            gray = gray.resize(expected_size, Image.BILINEAR)
        gray.save(alpha_dst_path)


def _prepare_mask_for_batch(mask_src_path, reference_frame_path, mask_dst_path):
    with Image.open(reference_frame_path) as ref_image:
        target_size = ref_image.size

    with Image.open(mask_src_path) as mask_image:
        gray = mask_image.convert("L")
        if gray.size != target_size:
            gray = gray.resize(target_size, Image.NEAREST)
        gray.save(mask_dst_path)


def _chunk_ranges(total_frames, chunk_size, overlap):
    if total_frames <= 0:
        return []
    chunk_size = max(1, chunk_size)
    overlap = max(0, min(overlap, chunk_size - 1))

    ranges = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + chunk_size)
        ranges.append((start, end))
        if end >= total_frames:
            break
        start = end - overlap
    return ranges


def _try_import_matte_sequence(config, projects_api, shot_uuid, slot_idx, final_alpha_dir, mode):
    if mode == "manual":
        return False, f"Manual import mode — alpha frames at: {final_alpha_dir}"

    alpha_files = sorted(
        f for f in os.listdir(final_alpha_dir)
        if f.lower().endswith((".png", ".jpg", ".tif", ".tiff", ".exr"))
    )
    if not alpha_files:
        if mode == "api-only":
            raise RuntimeError("No alpha frames found in output directory")
        return False, f"No alpha frames found in: {final_alpha_dir}"

    first_frame_path = os.path.normpath(os.path.join(final_alpha_dir, alpha_files[0]))

    # On SCRATCH 1.0.3 the move_shot (PATCH) endpoint is missing, so new versions
    # cannot be added to existing slots via REST.
    # Best available approach: add_shot creates the shot in the library,
    # then attempt move_shot — if it fails (expected on 1.0.3), fall through.
    from assimilate_client import ShotData as _ShotData, MoveShotData as _MoveShotData

    new_shot_uuid = None
    try:
        body = _ShotData()
        body.file = first_frame_path
        body.name = f"MatAnyone_matte_{shot_uuid[:8]}"
        new_shot = projects_api.add_shot(body=body)
        new_shot_uuid = getattr(new_shot, "uuid", None)
        if not new_shot_uuid:
            raise RuntimeError("add_shot returned no UUID")
        _print_step(f"Created matte shot in library: {new_shot_uuid}")
    except Exception as create_error:
        if mode == "api-only":
            raise RuntimeError(f"add_shot failed: {create_error}")
        _print_step(f"add_shot unavailable ({create_error.__class__.__name__}): {create_error}")

    if new_shot_uuid:
        try:
            # Get the count of existing versions to find the next index.
            versions_data = projects_api.get_construct_current_slot_versions(slot_idx=int(slot_idx))
            existing = getattr(versions_data, "shots", None) or []
            next_version_idx = len(existing)

            move_body = _MoveShotData()
            move_body.slot_idx = int(slot_idx)
            move_body.version_idx = next_version_idx
            move_body.create_copy = False
            projects_api.move_shot(body=move_body, shot_uuid=new_shot_uuid)
            return True, (
                f"Matte added to slot {slot_idx} version {next_version_idx} "
                f"(uuid={new_shot_uuid})"
            )
        except Exception as move_error:
            _print_step(
                f"move_shot not available on this server build "
                f"({move_error.__class__.__name__}). Trying add-shot-layer fallback."
            )

            try:
                layers_before = _get_shot_layers(projects_api, shot_uuid)
                layer_name = "MatAnyone_Matte"
                matte_payload = {
                    "shot_uuid": new_shot_uuid,
                    "blend_mode": "Copy",
                    "blur": 0,
                    "blur_angle": 0,
                    "blur_dir_active": False,
                    "channel_mask": {"R": True, "G": True, "B": True, "A": True},
                    "map": "Projected",
                    "slip": 0,
                    "orientation": {"rot": 0, "sx": 1, "sy": 1, "tx": 0, "ty": 0},
                    "warp": "None",
                    "warp_blur_filter": 0,
                    "warp_equi_dist": 0,
                }
                layer_payload = {
                    "name": layer_name,
                    "active": True,
                    "group": False,
                    "matte": matte_payload,
                }

                create_response = _api_post_json(config, f"/shot/{shot_uuid}/layers/new", layer_payload)

                def _field(obj, key, default=None):
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                    return getattr(obj, key, default)

                def _layer_part_uuid(layer_obj, part_name):
                    part = _field(layer_obj, part_name, {})
                    return _field(part, "shot_uuid", None)

                new_layer_idx = None
                with contextlib.suppress(Exception):
                    response_json = create_response.json()
                    for key in ("layer_idx", "index", "idx"):
                        candidate = response_json.get(key)
                        if isinstance(candidate, int):
                            new_layer_idx = candidate
                            break

                layers_after = _get_shot_layers(projects_api, shot_uuid)
                if new_layer_idx is None:
                    for idx in range(len(layers_after) - 1, -1, -1):
                        layer = layers_after[idx]
                        if str(_field(layer, "name", "")).strip() != layer_name:
                            continue
                        matte_uuid = _layer_part_uuid(layer, "matte")
                        if matte_uuid == new_shot_uuid:
                            new_layer_idx = idx
                            break

                if new_layer_idx is None:
                    if len(layers_after) > len(layers_before):
                        new_layer_idx = len(layers_before)
                    elif layers_after:
                        new_layer_idx = len(layers_after) - 1
                    else:
                        raise RuntimeError("Could not resolve created layer index")

                _print_step(f"Resolved MatAnyone_Matte layer index: {new_layer_idx}")

                # Ensure Fill is empty/reset: this layer should carry matte only.
                try:
                    requests.delete(
                        f"{config.host}/shot/{shot_uuid}/layers/{new_layer_idx}/fill",
                        timeout=20,
                    ).raise_for_status()
                except Exception:
                    pass

                _api_put_json(config, f"/shot/{shot_uuid}/layers/{new_layer_idx}/matte", matte_payload)

                # Read back matte assignment to verify SCRATCH persisted it.
                matte_readback = requests.get(
                    f"{config.host}/shot/{shot_uuid}/layers/{new_layer_idx}/matte",
                    timeout=20,
                )
                matte_readback.raise_for_status()
                persisted_matte_uuid = matte_readback.json().get("shot_uuid")
                if persisted_matte_uuid != new_shot_uuid:
                    _print_step(
                        "Layer matte did not persist after set-shot-layer-matte; "
                        "retrying with full set-shot-layer payload"
                    )
                    _api_put_json(
                        config,
                        f"/shot/{shot_uuid}/layers/{new_layer_idx}",
                        {
                            "name": layer_name,
                            "active": True,
                            "group": False,
                            "matte": matte_payload,
                        },
                    )
                    # Keep Fill cleared after full-layer update.
                    with contextlib.suppress(Exception):
                        requests.delete(
                            f"{config.host}/shot/{shot_uuid}/layers/{new_layer_idx}/fill",
                            timeout=20,
                        ).raise_for_status()

                    matte_readback = requests.get(
                        f"{config.host}/shot/{shot_uuid}/layers/{new_layer_idx}/matte",
                        timeout=20,
                    )
                    matte_readback.raise_for_status()
                    persisted_matte_uuid = matte_readback.json().get("shot_uuid")

                if persisted_matte_uuid != new_shot_uuid:
                    raise RuntimeError(
                        "Layer matte assignment mismatch after write-back "
                        f"(expected {new_shot_uuid}, got {persisted_matte_uuid})"
                    )

                return True, (
                    f"Matte linked as layer {new_layer_idx} on shot {shot_uuid} "
                    f"using source shot {new_shot_uuid}"
                )
            except Exception as layer_error:
                _print_step(
                    f"add-shot-layer fallback failed ({layer_error.__class__.__name__}): {layer_error}"
                )
                return False, (
                    f"Shot created in library (uuid={new_shot_uuid}) but could not be placed "
                    f"automatically (server 1.0.3 limitation). "
                    f"First alpha frame: {first_frame_path}"
                )

    return False, (
        f"Import not supported on this server build. "
        f"Drag the sequence into SCRATCH manually. "
        f"First frame: {first_frame_path}"
    )


def _write_shot_note(config, shot_uuid, note_text, status):
    endpoint = f"{config.host}/shot/{shot_uuid}"
    response = requests.get(endpoint, timeout=20)
    response.raise_for_status()
    payload = response.json()

    notes = payload.get("notes") or []
    notes.append(
        {
            "text": note_text,
            "note": note_text,
            "status": int(status),
            "frame": 0,
        }
    )
    payload["notes"] = notes

    put_response = requests.put(endpoint, json=payload, timeout=20)
    put_response.raise_for_status()

# =========================================================================
# --- MASK LAYER MANAGEMENT (SCRATCH Human Workflow Automation) ----------
# =========================================================================

def _disable_all_layers_except(config, shot_uuid, layers, keep_idx):
    """Disable all layers except the one at keep_idx."""
    for idx in range(len(layers)):
        if idx == keep_idx:
            continue
        layer = layers[idx]
        if getattr(layer, "group", False):
            continue  # Skip group layers
        try:
            _set_layer_active(config, shot_uuid, idx, False)
        except Exception as e:
            _print_step(f"Warning: could not disable layer {idx}: {e}")


def _restore_all_layers(config, shot_uuid, layer_states):
    """Restore layers to their original active states."""
    for idx, was_active in layer_states.items():
        try:
            _set_layer_active(config, shot_uuid, idx, was_active)
        except Exception as e:
            _print_step(f"Warning: could not restore layer {idx}: {e}")


def _save_layer_states(layers):
    """Save active state of all layers."""
    return {idx: bool(getattr(layer, "active", False)) for idx, layer in enumerate(layers)}


def _create_dedicated_mask_layer(config, shot_uuid, layer_name="MatAnyone_Mask"):
    """Create a new dedicated layer for the garbage mask.
    
    Mirrors the human workflow:
    1. Create a new layer where we will draw the garbage mask
    2. Configure it with brightness=0 (via colorgrade), invert (via canvas), 
       and matte blend mode = Subtract
    """
    layer_payload = {
        "name": layer_name,
        "active": True,
        "group": False,
    }
    
    create_response = _api_post_json(config, f"/shot/{shot_uuid}/layers/new", layer_payload)
    
    # Find the newly created layer index
    layers_after = _get_shot_layers_from_config(config, shot_uuid)
    new_layer_idx = None
    for idx in range(len(layers_after) - 1, -1, -1):
        if str(getattr(layers_after[idx], "name", "")).strip() == layer_name:
            new_layer_idx = idx
            break
    
    if new_layer_idx is None and layers_after:
        new_layer_idx = len(layers_after) - 1
    
    if new_layer_idx is None:
        raise RuntimeError("Could not find newly created mask layer")
    
    _print_step(f"Created mask layer '{layer_name}' at index {new_layer_idx}")
    return new_layer_idx


def _configure_mask_layer_for_matte_render(config, shot_uuid, layer_idx):
    """Configure the mask layer for proper matte rendering.
    
    Human workflow:
    - Set brightness to 0 (color-b.l=0)
    - Canvas invert on (canvas.alpha=0 in REST API)
    - Matte blend mode = Subtract (so mask area becomes transparent in alpha)
    
    All settings are applied in a single PUT to ensure atomicity.
    """
    # Single PUT with all layer properties
    layer_payload = {
        "name": "MatAnyone_Mask",
        "active": True,
        "group": False,
        "colorgrade": {
            "offset":     {"r": 0, "g": 0, "b": 0, "m": 0},
            "pre_gain":   {"r": 0, "g": 0, "b": 0, "m": 0},
            "color-a":    {"h": 0, "l": 0, "s": 0},
            "lift":       {"r": 0, "g": 0, "b": 0, "m": 0},
            "gamma":      {"r": 1, "g": 1, "b": 1, "m": 1},
            "gain":       {"r": 0, "g": 0, "b": 0, "m": 0},
            "color-b":    {"h": 0, "l": 0, "s": 0},       # brightness=0
            "tone":       {"c": 0, "i": 0, "s": 0},
            "temperature": {"k": 0, "t": 0},
            "aperature":  {"c": 0, "d": 0},
            "noise": 0,
            "channel_invert": False,
            "channel_remap": "^RRRA$",
            "soft_clip_high": {"r": 0, "g": 0, "b": 0},
            "soft_clip_low":  {"r": 0, "g": 0, "b": 0},
            "clip_high":      {"r": 0, "g": 0, "b": 0},
            "clip_low":       {"r": 0, "g": 0, "b": 0},
            "lut": "",
        },
        "canvas": {
            "alpha": 0,           # Canvas invert (alpha=0 = inverted)
            "softness": 0,
            "xform": {
                "pivot": {"x": 0, "y": 0, "z": 0},
                "rotate": {"x": 0, "y": 0, "z": 0},
                "scale": {"x": 1, "y": 1, "z": 1},
                "translate": {"x": 0, "y": 0, "z": 0},
            },
        },
        "matte": {
            "blend_mode": "Subtract",
            "blur": 0,
            "blur_angle": 0,
            "blur_dir_active": False,
            "channel_mask": {"R": True, "G": True, "B": True, "A": True},
            "map": "Projected",
            "slip": 0,
            "orientation": {"rot": 0, "sx": 1, "sy": 1, "tx": 0, "ty": 0},
            "warp": "None",
            "warp_blur_filter": 0,
            "warp_equi_dist": 0,
        },
    }
    try:
        _api_put_json(config, f"/shot/{shot_uuid}/layers/{layer_idx}", layer_payload)
        _print_step(
            f"Layer {layer_idx} configured: color-b.l=0, canvas.alpha=0 (invert), "
            f"matte=Subtract, channel_remap=RRRA"
        )
    except Exception as e:
        _print_step(f"Warning: could not configure layer {layer_idx}: {e}")

    # Clear any fill on this layer (it should carry matte only)
    try:
        requests.delete(
            f"{config.host}/shot/{shot_uuid}/layers/{layer_idx}/fill",
            timeout=20,
        ).raise_for_status()
    except Exception:
        pass


def _get_shot_layers_from_config(config, shot_uuid):
    """Get shot layers using raw HTTP (for when we don't have projects_api)."""
    response = requests.get(f"{config.host}/shot/{shot_uuid}/layers", timeout=20)
    response.raise_for_status()
    data = response.json()
    return list(getattr(data, "layers", []) or data.get("layers", []))


def _configure_output_for_rgba_tiff(config, output_uuid):
    """Configure the output node to render 16-bit TIFF RGBA.
    
    This is critical: the alpha channel in the rendered TIFF carries the
    actual matte shape from the Subtract blend mode.
    
    ShotDataOutput fields:
    - outputpath: str (output folder)
    - filespec: str (file specification mask)
    - format: int (format bitmask)
    - components: int (3=RGB, 4=RGBA)
    - extention: str (file extension)
    """
    endpoint = f"{config.host}/constructs/current/outputs/{output_uuid}"
    response = requests.get(endpoint, timeout=20)
    response.raise_for_status()
    payload = response.json()
    
    output_block = payload.setdefault("output", {})
    
    # Save original format for restoration later
    original_format = {
        "format": output_block.get("format"),
        "components": output_block.get("components"),
        "extention": output_block.get("extention"),
    }
    
    # Set RGBA (4 components) TIFF output
    # components=4 ensures RGBA so the alpha channel carries the matte
    # extention enum values: "tif" "dpx" "cin" "jpg" "tga" "j2c" "png" "exr" (no dot)
    output_block["components"] = 4        # RGBA
    output_block["extention"] = "tif"     # TIFF format (no dot prefix - API enum)
    
    update_response = requests.put(endpoint, json=payload, timeout=20)
    update_response.raise_for_status()
    
    _print_step(f"Output node configured: RGBA TIFF (components=4, ext=tif)")
    return original_format


def _restore_output_format(config, output_uuid, original_format):
    """Restore the output node to its original format settings."""
    endpoint = f"{config.host}/constructs/current/outputs/{output_uuid}"
    response = requests.get(endpoint, timeout=20)
    response.raise_for_status()
    payload = response.json()
    
    output_block = payload.setdefault("output", {})
    
    if original_format.get("format") is not None:
        output_block["format"] = original_format["format"]
    if original_format.get("components") is not None:
        output_block["components"] = original_format["components"]
    if original_format.get("extention") is not None:
        output_block["extention"] = original_format["extention"]
    
    requests.put(endpoint, json=payload, timeout=20)
    _print_step("Output node format restored to original settings")


def _extract_alpha_from_rgba(source_path, target_path):
    """Extract the alpha channel from an RGBA TIFF image.
    
    In the rendered RGBA TIFF with the mask layer using Subtract blend:
    - The ALPHA channel carries the matte shape
    - Painted (white mask) areas: alpha = 0 (transparent)
    - Unpainted areas: alpha = 1-255 (opaque)
    
    We invert the alpha to get the standard matte convention:
    - White (255) = foreground (the masked/painted area)
    - Black (0) = background
    
    Fallback: if alpha is all white (no transparency detected), try
    the red channel or luminance instead.
    """
    with Image.open(source_path) as image:
        _print_step(f"Rendered image mode: {image.mode}, size: {image.size}")
        
        # Ensure we have an alpha channel
        if image.mode == "RGBA":
            r, g, b, a = image.split()
            r_min, r_max = r.getextrema()
            g_min, g_max = g.getextrema()
            b_min, b_max = b.getextrema()
            a_min, a_max = a.getextrema()
            _print_step(f"Channel ranges: R=({r_min},{r_max}) G=({g_min},{g_max}) B=({b_min},{b_max}) A=({a_min},{a_max})")
            
            alpha_min, alpha_max = a_min, a_max
            
            if alpha_min == alpha_max == 255:
                # Alpha is all white (no transparency) - the Subtract blend
                # may not have produced meaningful alpha. Try fallback approaches.
                _print_step("Alpha is fully opaque (255). Trying red channel fallback...")
                red_min, red_max = r.getextrema()
                _print_step(f"Red channel range: min={red_min}, max={red_max}")
                
                if red_min != red_max:
                    # Red channel has variation - use it as the matte
                    from PIL import ImageOps
                    matte = ImageOps.invert(r)
                    matte.save(target_path)
                    _print_step(f"Used inverted red channel as matte: {target_path}")
                    return
                
                # Try luminance
                _print_step("Trying luminance fallback...")
                gray = image.convert("L")
                gray_min, gray_max = gray.getextrema()
                _print_step(f"Luminance range: min={gray_min}, max={gray_max}")
                gray.save(target_path)
                _print_step(f"Used luminance as matte: {target_path}")
                return
            
            # Alpha has variation - use it (this is the expected path)
            from PIL import ImageOps
            matte = ImageOps.invert(a)
            matte.save(target_path)
            _print_step(f"Alpha channel extracted and inverted: {target_path}")
            return
        
        # No alpha channel at all - try luminance
        _print_step(f"No alpha channel (mode={image.mode}). Using luminance fallback...")
        gray = image.convert("L")
        gray.save(target_path)
        _print_step(f"Used luminance as matte: {target_path}")


def _delete_layer(config, shot_uuid, layer_idx):
    """Delete a layer from the shot."""
    try:
        requests.delete(
            f"{config.host}/shot/{shot_uuid}/layers/{layer_idx}",
            timeout=20,
        ).raise_for_status()
        _print_step(f"Deleted layer {layer_idx}")
    except Exception as e:
        _print_step(f"Warning: could not delete layer {layer_idx}: {e}")


# =========================================================================
# --- UPDATED PASS B: CORRECT MATTE EXTRACTION ---------------------------
# =========================================================================

def _render_mask_pass_correct(config, app_api, projects_api, shot_uuid, output_uuid, mask_render_dir, cache_dir):
    """Render the mask using the correct human workflow:
    
    1. Create a dedicated mask layer (MatAnyone_Mask)
    2. Configure mask layer: brightness=0 (color-b.l=0), matte blend=Subtract
    3. Configure output node: RGBA TIFF
    4. Render ONE frame with ALL layers active (the Subtract blend needs
       the source plate below it to create meaningful alpha transparency)
    5. Extract alpha channel from the rendered RGBA TIFF
    6. Clean up: delete mask layer, restore output format
    
    Returns: path to the extracted matte PNG
    """
    mask_path = os.path.join(cache_dir, "mask.png")
    
    # Get current output format for restoration
    endpoint = f"{config.host}/constructs/current/outputs/{output_uuid}"
    response = requests.get(endpoint, timeout=20)
    response.raise_for_status()
    original_output = response.json()
    original_format = {
        "format": original_output.get("output", {}).get("format"),
        "components": original_output.get("output", {}).get("components"),
        "extention": original_output.get("output", {}).get("extention"),
    }
    
    new_layer_idx = None
    try:
        # Step 1: Create dedicated mask layer
        _print_step("Creating dedicated mask layer...")
        new_layer_idx = _create_dedicated_mask_layer(config, shot_uuid, "MatAnyone_Mask")
        
        # Step 2: Configure the mask layer (brightness=0, matte=Subtract)
        # NOTE: We do NOT disable other layers. The Subtract blend mode needs
        # the source plate (and other layers) below it to create meaningful
        # alpha transparency. The mask's white (painted) areas SUBTRACT from
        # the composite alpha, making those areas transparent.
        _print_step("Configuring mask layer (color-b.l=0, matte=Subtract)...")
        _configure_mask_layer_for_matte_render(config, shot_uuid, new_layer_idx)
        
        # Step 3: Configure output for RGBA TIFF
        _print_step("Configuring output for RGBA TIFF (components=4)...")
        _configure_output_for_rgba_tiff(config, output_uuid)
        
        # Step 4: Render one frame with ALL layers active
        _print_step("Rendering mask frame (full composite, RGBA TIFF)...")
        configure_current_output(config, output_uuid, mask_render_dir, single_frame=True)
        mask_queue_item, mask_render_started_at = _render_output_pass(
            app_api, output_uuid, expected_min_files=1
        )
        
        mask_source_dir = _render_item_path(mask_queue_item) or mask_render_dir
        
        # Find the rendered file
        generated_files = recent_render_files(
            [mask_render_dir, mask_source_dir],
            mask_render_started_at,
            (".tif", ".tiff", ".png"),
        )
        if not generated_files:
            generated_files = recent_render_files(
                [mask_render_dir, mask_source_dir],
                0,
                (".tif", ".tiff", ".png"),
            )
        if not generated_files:
            raise RuntimeError("Mask render produced no output files")
        
        rendered_file = generated_files[0]
        _print_step(f"Mask rendered: {rendered_file}")
        
        # Step 5: Extract alpha channel from RGBA TIFF
        _print_step("Extracting alpha channel from RGBA render...")
        _extract_alpha_from_rgba(rendered_file, mask_path)
        
        _print_step(f"Mask extraction complete: {mask_path}")
        
    finally:
        # Step 6: Clean up - restore everything
        _print_step("Restoring layer states and output format...")
        
        # Delete the mask layer we created
        if new_layer_idx is not None:
            _delete_layer(config, shot_uuid, new_layer_idx)
        
        # Restore output format
        _restore_output_format(config, output_uuid, original_format)
        
        _print_step("Cleanup complete")
    
    return mask_path


# =========================================================================
# --- STEP 2: VRAM LIMIT CALCULATOR & PRE-FLIGHT CHECKS ------------------
# =========================================================================
def get_safe_batch_limit():
    """
    Queries CUDA runtime for free VRAM bytes.
    Computes maximum frame batch windows dynamically to guarantee no OOM.
    """
    if not torch.cuda.is_available():
        return 8

    torch.cuda.empty_cache()
    free_vram, _ = torch.cuda.mem_get_info(device=0)
    free_vram_gb = free_vram / (1024 ** 3)
    
    # Keep 3GB reserved for base model weights and desktop UI threads
    usable_vram_mb = (free_vram_gb - 3.0) * 1024
    
    # 45MB represents the estimated deep layer feature footprint size per frame at 1080p
    max_frames = int(usable_vram_mb / 45.0)  
    
    # Clamp processing chunk sizes to prevent edge-case pipeline crashes
    return max(16, min(max_frames, 120))


def get_inference_device(require_cuda=False):
    cuda_available = bool(torch.cuda.is_available())
    cuda_build = torch.version.cuda
    device_count = int(torch.cuda.device_count()) if cuda_available else 0

    if cuda_available and device_count > 0:
        return "cuda:0"

    reason = (
        f"CUDA unavailable (torch.version.cuda={cuda_build}, "
        f"cuda_available={cuda_available}, device_count={device_count})"
    )
    if require_cuda:
        raise RuntimeError(f"{reason}. Install a CUDA-enabled PyTorch build in the runtime environment.")

    _print_step(f"Warning: {reason}. Falling back to CPU inference.")
    return "cpu"


# =========================================================================
# --- MAIN PIPELINE INTERACTION ENGINE ------------------------------------
# =========================================================================
def run_option1_pipeline(args):
    # 1. Setup API Connection Configuration pointing to APIV2
    config = Configuration()
    config.host = f"{args.scratch_host.rstrip('/')}/APIV2"
    
    api_client = ApiClient(config)
    system_api = SystemApi(api_client)
    projects_api = ProjectsApi(api_client)
    app_api = ApplicationApi(api_client)
    
    try:
        server_version = system_api.get_system_properties().rest_version
        _print_step(f"Connected to Assimilate REST V2, server version: {server_version}")
        log_version_support(server_version)

        shot = resolve_target_shot(projects_api, app_api, args)
        shot_uuid = shot.uuid
        _print_step(f"Using cache directory: {args.cache_dir}")

    except ApiException as e:
        _print_step(f"Handshake failed: {e.body.decode('utf-8') if e.body else str(e)}")
        return None

    # Build workspace directories mapping to the shot's unique identifier
    workspace_dir = os.path.join(args.cache_dir, shot_uuid)
    export_dir = os.path.join(workspace_dir, "source_frames")
    output_dir = os.path.join(workspace_dir, "results")
    mask_render_dir = os.path.join(workspace_dir, "mask_render")

    # Final matte output goes to SCRATCH project render folder
    try:
        proj_paths = projects_api.get_projects_current()
        render_path = getattr(proj_paths, "project_paths", None)
        if render_path:
            render_path = getattr(render_path, "render_path", None)
    except Exception:
        render_path = None

    # Get construct name and slot index for folder structure
    try:
        construct_data = projects_api.get_constructs_current(level="ALL")
        construct_name = getattr(construct_data, "name", None) or "Unknown"
    except Exception:
        construct_name = "Unknown"
    try:
        selected = projects_api.get_construct_current_selected_shots(level="ALL")
        selections = _extract_selected_entries(selected)
        slot_idx = int(getattr(selections[0], "slot_idx", 0) or 0) if selections else 0
    except Exception:
        slot_idx = 0

    if render_path and os.path.isdir(render_path):
        shot_label = getattr(shot, "name", None) or shot_uuid[:8]
        final_alpha_dir = os.path.join(
            render_path, "AiMatte", construct_name, f"{slot_idx:03d}_{shot_label}", "Alphas", shot_label, shot_label
        )
    else:
        final_alpha_dir = os.path.join(output_dir, "alpha")
    
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_alpha_dir, exist_ok=True)
    os.makedirs(mask_render_dir, exist_ok=True)

    _print_step("Creating dedicated MatAnyone output node in project media folder")
    try:
        output_uuid, export_dir = create_matanyone_output(
            config, projects_api, shot_uuid,
            int(getattr(shot, "length", 0) or 0),
            getattr(shot, "name", None) or shot_uuid[:8],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create MatAnyone output node: {e}")

    # Read the actual render output path from the output node
    output_node_name = None
    try:
        out_resp = requests.get(f"{config.host}/constructs/current/outputs/{output_uuid}", timeout=20)
        out_resp.raise_for_status()
        out_data = out_resp.json()
        actual_output_path = out_data.get("output", {}).get("outputpath", "")
        output_node_name = out_data.get("name", "")
        if actual_output_path and os.path.isdir(actual_output_path):
            render_dir = actual_output_path
        else:
            render_dir = export_dir
    except Exception:
        render_dir = export_dir

    # The output node's filespec creates a subfolder named after the output.
    # Search for frames in that specific subfolder, not the entire render directory.
    if output_node_name:
        output_subfolder = os.path.join(render_dir, output_node_name)
        if os.path.isdir(output_subfolder):
            render_dir = output_subfolder

    # =========================================================================
    # PASS A: Render clean plates (ALL layers disabled for raw source)
    # =========================================================================
    layers = _get_shot_layers(projects_api, shot_uuid)

    expected_plate_count = int(getattr(shot, "length", 0) or 0)

    # Check if rendered frames already exist from a previous run
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".dpx", ".exr")
    existing_frames = recent_render_files([render_dir], 0, exts)
    _print_step(f"Raw frames found in {render_dir}: {len(existing_frames)}")
    if existing_frames:
        _print_step(f"  First: {existing_frames[0]}")
    # Exclude frames from any AiMatte subdirectory (matte outputs)
    existing_frames = [f for f in existing_frames
                       if "\\aimatte\\" not in os.path.normcase(os.path.normpath(f).replace("/", "\\"))]
    _print_step(f"After excluding AiMatte: {len(existing_frames)} frames")
    if existing_frames:
        _print_step(f"  First: {existing_frames[0]}")
    render_source_dir = render_dir

    # Check SCRATCH render queue for this output node (queue UUID != output UUID)
    output_rendered = False
    try:
        queue_list_resp = requests.get(f"{config.host}/application/render", timeout=10)
        if queue_list_resp.status_code == 200:
            queue_items = queue_list_resp.json()
            if isinstance(queue_items, list):
                for item in queue_items:
                    item_name = str(item.get("name", ""))
                    item_status = str(item.get("status", "")).lower()
                    if item_name.startswith("MatAnyone_") and item_status in ("finished", "complete"):
                        output_rendered = True
                        _print_step(f"Output render found in queue: [{item_name}] status={item_status}")
                        break
    except Exception:
        pass

    if output_rendered:
        if expected_plate_count > 0:
            all_frames = existing_frames[:expected_plate_count]
        else:
            all_frames = existing_frames
        _print_step(f"PASS A skipped: output rendered, {len(all_frames)} frames in {render_dir}")
    else:
        if expected_plate_count > 0:
            _print_step(f"PASS A: rendering clean plates 1/{expected_plate_count}..{expected_plate_count}/{expected_plate_count}")
        else:
            _print_step("PASS A: rendering clean plate sequence with ALL layers disabled")
        try:
            # Disable ALL layers so the clean plate is the raw source footage
            layer_states_before_pass_a = _save_layer_states(layers)
            for idx in range(len(layers)):
                try:
                    _set_layer_active(config, shot_uuid, idx, False)
                except Exception:
                    pass
            _print_step(f"Disabled all {len(layers)} layers for clean plate render")

            configure_current_output(config, output_uuid, export_dir, single_frame=False)
            queue_item, render_started_at = _render_output_pass(
                app_api,
                output_uuid,
                expected_min_files=max(1, expected_plate_count),
            )
            render_source_dir = _render_item_path(queue_item) or export_dir
        except Exception as e:
            raise RuntimeError(f"Failed during clean plate render: {e}")
        finally:
            # Restore all layers to their original states
            _restore_all_layers(config, shot_uuid, layer_states_before_pass_a)
            _print_step("Restored all layer states after clean plate render")

        # ── Collect rendered frames ──
        _print_step("Collecting rendered clean plate frames")
        all_frames = recent_render_files(
            [render_source_dir, export_dir], render_started_at, exts,
        )
        if not all_frames:
            all_frames = recent_render_files(
                [render_source_dir, export_dir], 0, exts,
            )

    total_frames = len(all_frames)
    if total_frames == 0:
        raise RuntimeError(f"No clean plate frames found in: {export_dir}")
    frames_per_chunk = get_safe_batch_limit()
    device = get_inference_device(require_cuda=args.require_cuda)

    # ── PASS B: Interactive SAM2 mask editor ────────────────────────────────
    mask_path = os.path.join(args.cache_dir, "mask.png")
    if args.skip_sam and os.path.isfile(mask_path):
        _print_step(f"PASS B skipped (--skip-sam): reusing existing mask → {mask_path}")
    else:
        _print_step("PASS B: Launching SAM2 interactive mask editor")
        _print_step(f"  Source frame : {all_frames[0]}")
        _print_step(f"  Mask output  : {mask_path}")
        _print_step("A browser window will open. Click on subjects, then click 'Save & Continue'.")

        _editor_dir = os.path.dirname(os.path.abspath(__file__))
        if _editor_dir not in sys.path:
            sys.path.insert(0, _editor_dir)
        from sam_mask_editor import launch_mask_editor  # noqa: PLC0415

        launch_mask_editor(
            frame_path=all_frames[0],
            output_mask_path=mask_path,
            device=device,
            port=args.sam_port,
            open_browser=not args.no_browser,
        )

        if not os.path.isfile(mask_path):
            raise RuntimeError(
                "The SAM2 editor closed without saving a mask. "
                "Re-run or use --skip-sam with an existing mask.png."
            )
        _print_step(f"PASS B complete: mask saved → {mask_path}")

    _print_step("Running MatAnyone2 with VRAM-based chunking")
    _print_step(f"MatAnyone inference device: {device}; chunk size: {frames_per_chunk}; overlap: {args.chunk_overlap}")

    _print_step("Loading MatAnyone2 weights")
    model = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2")
    # Keep fp32 weights for compatibility with MatAnyone2 internals on some CUDA paths.
    model = model.to(device)

    ranges = _chunk_ranges(total_frames, frames_per_chunk, args.chunk_overlap)
    active_mask_path = mask_path

    for chunk_idx, (current_start, current_end) in enumerate(ranges):
        _print_step(f"Processing chunk {chunk_idx + 1}/{len(ranges)}: frames {current_start}..{current_end - 1}")

        batch_input_dir = os.path.join(workspace_dir, f"batch_{current_start}_in")
        batch_output_dir = os.path.join(workspace_dir, f"batch_{current_start}_out")
        os.makedirs(batch_input_dir, exist_ok=True)
        os.makedirs(batch_output_dir, exist_ok=True)

        batch_frames = all_frames[current_start:current_end]
        original_sizes = _prepare_batch_input(batch_frames, batch_input_dir, args.max_min_side)

        processor = InferenceCore(model, device=device)
        batch_mask_path = active_mask_path
        if batch_frames:
            reference_frame = os.path.join(batch_input_dir, os.path.basename(batch_frames[0]))
            if os.path.isfile(reference_frame):
                prepared_mask_path = os.path.join(workspace_dir, f"mask_for_{current_start}.png")
                _prepare_mask_for_batch(active_mask_path, reference_frame, prepared_mask_path)
                batch_mask_path = prepared_mask_path

        autocast_context = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with autocast_context:
            processor.process_video(
                input_path=batch_input_dir,
                mask_path=batch_mask_path,
                output_path=batch_output_dir
            )

        chunk_alpha_src = os.path.join(batch_output_dir, "alpha")
        generated_alphas = []

        # MatAnyone2 may output video files instead of image sequences.
        # Detect *_pha.mp4 (alpha channel video) and extract frames to alpha/.
        pha_video = None
        for fname in os.listdir(batch_output_dir):
            if fname.lower().endswith(".mp4") and "_pha" in fname.lower():
                pha_video = os.path.join(batch_output_dir, fname)
                break

        if pha_video:
            import imageio
            os.makedirs(chunk_alpha_src, exist_ok=True)
            _print_step(f"Extracting alpha frames from video: {os.path.basename(pha_video)}")
            reader = imageio.get_reader(pha_video)
            for frame_idx, frame in enumerate(reader):
                frame_name = f"{current_start + frame_idx:06d}.png"
                frame_path = os.path.join(chunk_alpha_src, frame_name)
                Image.fromarray(frame).convert("L").save(frame_path)
            reader.close()
            _print_step(f"Extracted {frame_idx + 1} alpha frames to {chunk_alpha_src}")

        # Collect extracted or conventionally placed image frames.
        if os.path.isdir(chunk_alpha_src):
            generated_alphas = sorted(
                f for f in os.listdir(chunk_alpha_src)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"))
            )

        if not generated_alphas:
            tree = []
            for walk_root, _dirs, walk_files in os.walk(batch_output_dir):
                rel = os.path.relpath(walk_root, batch_output_dir)
                for fname in walk_files:
                    tree.append(f"  {rel}/{fname}")
            _print_step("batch_output_dir tree:\n" + ("\n".join(tree) if tree else "  (empty)"))
            raise RuntimeError(f"Inference produced no alpha outputs for chunk {chunk_idx + 1}")

        skip_count = args.chunk_overlap if chunk_idx > 0 else 0
        usable_alphas = generated_alphas[skip_count:] if skip_count < len(generated_alphas) else generated_alphas[-1:]

        for alpha_file in usable_alphas:
            alpha_source = os.path.join(chunk_alpha_src, alpha_file)
            alpha_target = os.path.join(final_alpha_dir, alpha_file)
            expected_size = original_sizes.get(alpha_file)
            if expected_size is None and batch_frames:
                fallback_name = os.path.basename(batch_frames[0])
                expected_size = original_sizes.get(fallback_name, (1920, 1080))
            _restore_alpha_size(alpha_source, alpha_target, expected_size)

        last_generated_alpha = os.path.join(chunk_alpha_src, generated_alphas[-1])
        active_mask_path = os.path.join(workspace_dir, f"prop_mask_{current_end}.png")
        shutil.copy(last_generated_alpha, active_mask_path)

        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            shutil.rmtree(batch_input_dir)
            shutil.rmtree(batch_output_dir)
        except Exception:
            pass

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Import matte back into SCRATCH
    # =========================================================================
    _print_step("Attempting to load generated matte back into SCRATCH")
    try:
        selected = projects_api.get_construct_current_selected_shots(level="ALL")
        selections = _extract_selected_entries(selected)
        shot_slot_idx = 0
        if selections:
            shot_slot_idx = int(getattr(selections[0], "slot_idx", 0) or 0)
    except Exception:
        shot_slot_idx = 0

    imported, import_message = _try_import_matte_sequence(
        config, projects_api, shot_uuid, shot_slot_idx, final_alpha_dir, args.import_mode
    )
    _print_step(import_message)

    note_text = (
        f"MatAnyone2 matte generated. shot={shot_uuid}; alpha_dir={final_alpha_dir}; "
        f"imported={imported}; mode={args.import_mode}"
    )
    try:
        _write_shot_note(config, shot_uuid, note_text, args.note_status)
        _print_step("Shot note written")
    except Exception as error:
        _print_step(f"Shot note write failed (non-fatal): {error}")

    if not args.keep_temp:
        keep_paths = [final_alpha_dir]
        _cleanup_temp_workspace(workspace_dir, keep_paths=keep_paths)
        _print_step(f"Temporary files cleaned. Preserved: {final_alpha_dir}")

    # Open the alpha folder in Explorer so the user can drag into SCRATCH if needed.
    try:
        import subprocess
        subprocess.Popen(f'explorer "{os.path.normpath(final_alpha_dir)}"')
    except Exception:
        pass

    _print_step("Pipeline execution completed")


if __name__ == "__main__":
    argument_parser = build_argument_parser()
    run_option1_pipeline(argument_parser.parse_args())



