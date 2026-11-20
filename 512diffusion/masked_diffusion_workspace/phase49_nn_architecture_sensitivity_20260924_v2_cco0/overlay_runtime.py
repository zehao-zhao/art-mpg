"""Verify and activate the immutable Phase-48 Python/native runtime."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import protocol


PHASE48_PROTOCOL_ID = "phase48_npm_style_ablation_20260924_v1"


class OverlayError(RuntimeError):
    """Raised when the bound Phase-48 runtime is incomplete or inconsistent."""


def _workspace(value: object) -> Path:
    return protocol.workspace_path(value, "Phase48 overlay path")


def _phase48_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    overlay = manifest["phase48_overlay"]
    path = _workspace(overlay["manifest"])
    if protocol.sha256_file(path) != overlay["manifest_sha256"]:
        raise OverlayError("bound Phase48 manifest changed")
    value = protocol.load_json(path)
    if value.get("protocol_id") != PHASE48_PROTOCOL_ID:
        raise OverlayError("bound overlay is not the Phase48 protocol")
    return value


def verify_python_overlay(manifest: dict[str, Any]) -> dict[str, Any]:
    overlay = manifest["phase48_overlay"]
    source_manifest = _phase48_manifest(manifest)
    source_overlay = source_manifest.get("experiment_overlay", {})
    for key in (
        "python_root", "python_package", "native_header_snapshot", "inventory",
        "inventory_sha256", "files_sha256", "experiment_plugin_dir",
        "experiment_mpg_plugin", "bootstrap_receipt",
    ):
        if source_overlay.get(key) != overlay.get(key):
            raise OverlayError(f"Phase49/Phase48 overlay binding differs: {key}")

    inventory_path = _workspace(overlay["inventory"])
    inventory = protocol.load_json(inventory_path)
    if (
        inventory.get("schema_version") != 1
        or inventory.get("protocol_id") != PHASE48_PROTOCOL_ID
        or inventory.get("status") != "overlay_frozen"
    ):
        raise OverlayError("Phase48 overlay inventory is missing or invalid")
    if protocol.sha256_file(inventory_path) != overlay["inventory_sha256"]:
        raise OverlayError("Phase48 overlay inventory differs from its binding")
    if inventory.get("files_sha256") != overlay["files_sha256"]:
        raise OverlayError("Phase48 overlay aggregate digest differs")

    overlay_root = _workspace(overlay["python_root"])
    files = inventory.get("files")
    if not isinstance(files, dict) or not files:
        raise OverlayError("Phase48 overlay inventory contains no files")
    observed = {
        path.relative_to(overlay_root).as_posix(): protocol.sha256_file(path)
        for path in sorted(overlay_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    if observed != files or protocol.canonical_sha256(files) != overlay["files_sha256"]:
        raise OverlayError("Phase48 overlay payload changed")
    package_root = _workspace(overlay["python_package"])
    header = _workspace(overlay["native_header_snapshot"])
    if not (package_root / "__init__.py").is_file() or not header.is_file():
        raise OverlayError("Phase48 overlay lacks its Python package or native header")
    return {
        "phase48_manifest_sha256": overlay["manifest_sha256"],
        "inventory_path": protocol.portable_workspace(inventory_path),
        "inventory_sha256": overlay["inventory_sha256"],
        "files_sha256": overlay["files_sha256"],
        "file_count": len(files),
    }


def activate_python_overlay(manifest: dict[str, Any]) -> dict[str, Any]:
    identity = verify_python_overlay(manifest)
    root = _workspace(manifest["phase48_overlay"]["python_root"])
    for name, module in list(sys.modules.items()):
        if name != "masked_diffusion_mpg" and not name.startswith("masked_diffusion_mpg."):
            continue
        origin = Path(str(getattr(module, "__file__", ""))).resolve()
        try:
            origin.relative_to(root.resolve())
        except ValueError as exc:
            raise OverlayError(
                f"non-Phase48 module imported before overlay activation: {name}: {origin}"
            ) from exc
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    package = importlib.import_module("masked_diffusion_mpg")
    origin = Path(str(package.__file__)).resolve()
    try:
        origin.relative_to(root.resolve())
    except ValueError as exc:
        raise OverlayError(f"failed to activate Phase48 Python overlay: {origin}") from exc
    return {**identity, "active_package": str(origin)}


def verify_native_runtime(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify both the Phase48 copy and the plugin Mitsuba actually resolves."""

    overlay = manifest["phase48_overlay"]
    receipt_path = _workspace(overlay["bootstrap_receipt"])
    receipt = protocol.load_json(receipt_path)
    expected = manifest["implementation"]["expected_global_mpg_plugin_sha256"]
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != PHASE48_PROTOCOL_ID
        or receipt.get("status") != "experiment_runtime_ready"
        or receipt.get("experiment_mpg_plugin_sha256") != expected
    ):
        raise OverlayError("Phase48 native-runtime receipt is missing or has the wrong plugin")
    phase48_plugin = _workspace(overlay["experiment_mpg_plugin"])
    global_plugin = protocol.project_path(manifest["implementation"]["mpg_plugin"])
    for label, path in (("Phase48 plugin", phase48_plugin), ("actual global plugin", global_plugin)):
        if protocol.sha256_file(path) != expected:
            raise OverlayError(f"{label} hash differs from the bound Phase48 plugin: {path}")
    return {
        "bootstrap_receipt_path": protocol.portable_workspace(receipt_path),
        "bootstrap_receipt_sha256": protocol.sha256_file(receipt_path),
        "phase48_plugin_path": protocol.portable_workspace(phase48_plugin),
        "actual_global_plugin_path": protocol.portable_project(global_plugin),
        "actual_global_plugin_sha256": expected,
    }


def install_phase20_runtime_override(source_worker: Any, manifest: dict[str, Any]) -> None:
    """Point Phase20 at the verified directory Mitsuba actually resolves."""

    verify_native_runtime(manifest)
    global_plugin = protocol.project_path(manifest["implementation"]["mpg_plugin"])
    plugin_dir = global_plugin.parent
    original = source_worker._runtime_profile

    def verified_runtime_profile() -> tuple[Any, Any]:
        phase13, profile = original()
        return phase13, replace(profile, plugin_dir=plugin_dir, plugin_text=str(plugin_dir))

    source_worker._runtime_profile = verified_runtime_profile
