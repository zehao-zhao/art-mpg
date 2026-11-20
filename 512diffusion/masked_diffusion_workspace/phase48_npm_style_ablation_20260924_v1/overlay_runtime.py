"""Validation and activation helpers for the isolated Phase-48 overlay."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import protocol


class OverlayError(RuntimeError):
    """Raised when the experiment overlay is incomplete or inconsistent."""


def _workspace(value: object) -> Path:
    return protocol.workspace_path(value, "overlay path")


def _project(value: object) -> Path:
    return protocol.project_path(value, "overlay project path")


def verify_python_overlay(manifest: dict[str, Any]) -> dict[str, Any]:
    overlay = manifest["experiment_overlay"]
    inventory_path = _workspace(overlay["inventory"])
    inventory = protocol.load_json(inventory_path)
    if (
        inventory.get("schema_version") != 1
        or inventory.get("protocol_id") != protocol.PROTOCOL_ID
        or inventory.get("status") != "overlay_frozen"
    ):
        raise OverlayError("Python/native overlay inventory is missing or invalid")
    if protocol.sha256_file(inventory_path) != overlay.get("inventory_sha256"):
        raise OverlayError("overlay inventory differs from the manifest binding")
    if inventory.get("files_sha256") != overlay.get("files_sha256"):
        raise OverlayError("overlay aggregate file hash differs from the manifest binding")
    package_root = _workspace(overlay["python_package"])
    header = _workspace(overlay["native_header_snapshot"])
    files = inventory.get("files")
    if not isinstance(files, dict) or not files:
        raise OverlayError("overlay inventory contains no files")
    observed: dict[str, str] = {}
    overlay_root = _workspace(overlay["python_root"])
    for path in sorted(overlay_root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            observed[path.relative_to(overlay_root).as_posix()] = protocol.sha256_file(path)
    if observed != files:
        raise OverlayError("experiment overlay differs from its frozen inventory")
    if protocol.canonical_sha256(files) != overlay.get("files_sha256"):
        raise OverlayError("experiment overlay aggregate digest changed")
    if not (package_root / "__init__.py").is_file() or not header.is_file():
        raise OverlayError("overlay lacks the full Python package or native header")
    return {
        "inventory_path": protocol.portable_workspace(inventory_path),
        "inventory_sha256": protocol.sha256_file(inventory_path),
        "files_sha256": protocol.canonical_sha256(files),
        "file_count": len(files),
    }


def activate_python_overlay(manifest: dict[str, Any]) -> dict[str, Any]:
    identity = verify_python_overlay(manifest)
    root = _workspace(manifest["experiment_overlay"]["python_root"])
    foreign = [
        (name, module) for name, module in sys.modules.items()
        if name == "masked_diffusion_mpg" or name.startswith("masked_diffusion_mpg.")
    ]
    for name, module in foreign:
        origin = Path(str(getattr(module, "__file__", ""))).resolve()
        try:
            origin.relative_to(root.resolve())
        except ValueError as exc:
            raise OverlayError(f"production module was imported before overlay activation: {name}: {origin}") from exc
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    package = importlib.import_module("masked_diffusion_mpg")
    origin = Path(str(package.__file__)).resolve()
    try:
        origin.relative_to(root.resolve())
    except ValueError as exc:
        raise OverlayError(f"failed to activate experiment Python overlay: {origin}") from exc
    return {**identity, "active_package": str(origin)}


def verify_native_runtime(manifest: dict[str, Any]) -> dict[str, Any]:
    overlay = manifest["experiment_overlay"]
    receipt_path = _workspace(overlay["bootstrap_receipt"])
    receipt = protocol.load_json(receipt_path)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != protocol.PROTOCOL_ID
        or receipt.get("status") != "experiment_runtime_ready"
    ):
        raise OverlayError("native bootstrap receipt is missing or invalid")
    plugin_dir = _workspace(overlay["experiment_plugin_dir"])
    plugin = _workspace(overlay["experiment_mpg_plugin"])
    inventory = receipt.get("plugin_inventory")
    if not isinstance(inventory, dict) or not inventory:
        raise OverlayError("native bootstrap plugin inventory is empty")
    observed = {
        path.relative_to(plugin_dir).as_posix(): protocol.sha256_file(path)
        for path in sorted(plugin_dir.rglob("*")) if path.is_file()
    }
    if observed != inventory:
        raise OverlayError("experiment plugin directory differs from bootstrap receipt")
    if receipt.get("experiment_mpg_plugin_sha256") != protocol.sha256_file(plugin):
        raise OverlayError("experiment MPG plugin hash changed")
    return {
        "bootstrap_receipt_path": protocol.portable_workspace(receipt_path),
        "bootstrap_receipt_sha256": protocol.sha256_file(receipt_path),
        "plugin_dir": protocol.portable_workspace(plugin_dir),
        "plugin_inventory_sha256": protocol.canonical_sha256(inventory),
        "experiment_mpg_plugin_sha256": receipt["experiment_mpg_plugin_sha256"],
    }


def install_phase20_runtime_override(source_worker: Any, manifest: dict[str, Any]) -> None:
    """Point Phase20 rendering at the experiment-local plugin directory."""
    native = verify_native_runtime(manifest)
    plugin_dir = protocol.workspace_path(native["plugin_dir"])
    original = source_worker._runtime_profile

    def isolated_runtime_profile() -> tuple[Any, Any]:
        phase13, profile = original()
        isolated = replace(
            profile,
            plugin_dir=plugin_dir,
            plugin_text=str(plugin_dir),
        )
        return phase13, isolated

    source_worker._runtime_profile = isolated_runtime_profile
