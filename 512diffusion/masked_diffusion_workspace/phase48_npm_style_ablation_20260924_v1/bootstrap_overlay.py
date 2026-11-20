#!/usr/bin/env python3
"""Freeze Python/header inputs and build an isolated Phase-48 plugin tree."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import protocol  # noqa: E402
import overlay_runtime  # noqa: E402


class BootstrapError(RuntimeError):
    """Raised when an overlay/bootstrap operation cannot remain isolated."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path: Path, payload: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, protocol.json_bytes(value))


def freeze_overlay(manifest: dict[str, Any]) -> dict[str, Any]:
    overlay = manifest["experiment_overlay"]
    source_package = protocol.workspace_path(overlay["python_source"])
    target_root = protocol.workspace_path(overlay["python_root"])
    target_package = protocol.workspace_path(overlay["python_package"])
    target_header = protocol.workspace_path(overlay["native_header_snapshot"])
    inventory_path = protocol.workspace_path(overlay["inventory"])
    source_header = protocol.project_path(overlay["native_header_source"])

    if target_root.exists() or inventory_path.exists():
        identity = overlay_runtime.verify_python_overlay(manifest)
        frozen = protocol.load_json(inventory_path)["files"]
        current = {
            f"masked_diffusion_mpg/{path.relative_to(source_package).as_posix()}":
                protocol.sha256_file(path)
            for path in sorted(source_package.rglob("*.py"))
        }
        current["native/masked_diffusion_runtime.h"] = protocol.sha256_file(source_header)
        if current != frozen:
            raise BootstrapError(
                "source implementation differs from the frozen overlay; create a new protocol version"
            )
        print("existing overlay is valid; leaving it unchanged", file=sys.stderr)
        return {"status": "already_frozen", **identity}

    staging = target_root.with_name(f".{target_root.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise BootstrapError(f"staging path already exists: {staging}")
    try:
        for source in sorted(source_package.rglob("*.py")):
            relative = source.relative_to(source_package)
            destination = staging / "masked_diffusion_mpg" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        header_destination = staging / "native" / "masked_diffusion_runtime.h"
        header_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_header, header_destination)
        files = {
            path.relative_to(staging).as_posix(): protocol.sha256_file(path)
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        if not files or "masked_diffusion_mpg/__init__.py" not in files:
            raise BootstrapError("failed to stage the complete Python package")
        os.replace(staging, target_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    inventory = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "overlay_frozen",
        "source_python_package": overlay["python_source"],
        "source_native_header": overlay["native_header_source"],
        "files": files,
        "files_sha256": protocol.canonical_sha256(files),
        "frozen_at": now_utc(),
    }
    atomic_json(inventory_path, inventory)
    identity = overlay_runtime.verify_python_overlay(manifest)
    return {"status": "overlay_frozen", **identity}


def _tree_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): protocol.sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def build_native(manifest: dict[str, Any]) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise BootstrapError("native experiment plugin bootstrap is Linux-only")
    python_overlay = overlay_runtime.verify_python_overlay(manifest)
    overlay = manifest["experiment_overlay"]
    source_header = protocol.project_path(overlay["native_header_source"])
    frozen_header = protocol.workspace_path(overlay["native_header_snapshot"])
    build_dir = protocol.project_path(overlay["linux_build_dir"])
    built_plugin = protocol.project_path(overlay["built_mpg_plugin"])
    production_dir = protocol.project_path(overlay["production_plugin_dir"])
    production_plugin = protocol.project_path(overlay["production_mpg_plugin"])
    experiment_dir = protocol.workspace_path(overlay["experiment_plugin_dir"])
    receipt_path = protocol.workspace_path(overlay["bootstrap_receipt"])
    if experiment_dir.exists() or receipt_path.exists():
        identity = overlay_runtime.verify_native_runtime(manifest)
        print("existing experiment runtime is valid; leaving it unchanged", file=sys.stderr)
        return {"status": "already_ready", **identity}
    if not build_dir.is_dir() or not production_dir.is_dir():
        raise BootstrapError("Linux build or production plugin directory is missing")
    if source_header.is_symlink() or not source_header.is_file():
        raise BootstrapError("production MPG header must be a regular non-symlink file")

    original_header = source_header.read_bytes()
    original_header_mode = stat.S_IMODE(source_header.stat().st_mode)
    original_header_sha = protocol.sha256_file(source_header)
    production_plugin_sha = protocol.sha256_file(production_plugin)
    frozen_header_sha = protocol.sha256_file(frozen_header)
    runtime_root = experiment_dir.parents[1]
    staging_root = runtime_root.with_name(f".{runtime_root.name}.{os.getpid()}.tmp")
    staging_plugins = staging_root / "dist" / "plugins"
    build_log = PACKAGE_ROOT / "native_build.stdout.log"
    build_error = PACKAGE_ROOT / "native_build.stderr.log"
    old_handlers: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupted)
    try:
        atomic_bytes(source_header, frozen_header.read_bytes(), mode=original_header_mode)
        command = [
            "cmake", "--build", str(build_dir), "--target",
            "src/integrators/manifold_path_guiding.so", "--parallel", "95",
        ]
        with build_log.open("w", encoding="utf-8") as stdout, build_error.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr,
                text=True, check=False,
            )
        if completed.returncode != 0:
            raise BootstrapError(f"native target build failed with exit {completed.returncode}")
        if not built_plugin.is_file():
            raise BootstrapError(f"native build produced no plugin: {built_plugin}")
        if staging_root.exists():
            raise BootstrapError(f"staging runtime already exists: {staging_root}")
        shutil.copytree(production_dir, staging_plugins, symlinks=False)
        shutil.copy2(built_plugin, staging_plugins / "manifold_path_guiding.so")
        os.replace(staging_root, runtime_root)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    finally:
        atomic_bytes(source_header, original_header, mode=original_header_mode)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    if protocol.sha256_file(source_header) != original_header_sha:
        raise BootstrapError("production source header was not restored byte-for-byte")
    if protocol.sha256_file(production_plugin) != production_plugin_sha:
        raise BootstrapError("production dist plugin changed during isolated bootstrap")
    experiment_plugin = protocol.workspace_path(overlay["experiment_mpg_plugin"])
    plugin_inventory = _tree_inventory(experiment_dir)
    receipt = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "experiment_runtime_ready",
        "python_overlay": python_overlay,
        "source_header_original_sha256": original_header_sha,
        "source_header_overlay_sha256": frozen_header_sha,
        "source_header_restored_sha256": protocol.sha256_file(source_header),
        "production_mpg_plugin_sha256": production_plugin_sha,
        "experiment_mpg_plugin_sha256": protocol.sha256_file(experiment_plugin),
        "plugin_inventory": plugin_inventory,
        "plugin_inventory_sha256": protocol.canonical_sha256(plugin_inventory),
        "build_command": command,
        "build_stdout_sha256": protocol.sha256_file(build_log),
        "build_stderr_sha256": protocol.sha256_file(build_error),
        "built_at": now_utc(),
    }
    atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_sha256": protocol.sha256_file(receipt_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-overlay", "build-native", "verify"))
    args = parser.parse_args(argv)
    manifest = protocol.load_manifest()
    if args.command == "freeze-overlay":
        result = freeze_overlay(manifest)
    elif args.command == "build-native":
        result = build_native(manifest)
    else:
        result = {
            "protocol_id": protocol.PROTOCOL_ID,
            "python_overlay": overlay_runtime.verify_python_overlay(manifest),
            "native_runtime": overlay_runtime.verify_native_runtime(manifest),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BootstrapError, overlay_runtime.OverlayError, protocol.ProtocolError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
