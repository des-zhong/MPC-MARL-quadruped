"""Small runtime-only compatibility shims for the supported Isaac Lab baseline.

The migration targets Isaac Sim 5.1 with Isaac Lab v2.3.x.  Isaac Lab 2.3.x
pins the Isaac Sim 5.1 URDF importer to ``2.4.31``, while some Isaac Sim 5.1
pip distributions ship only the unversioned ``2.4.30`` extension.  This module
keeps that packaging mismatch local to the migrated project; it is a no-op in
the normal matched installation and is never imported by simulator-free tests.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_URDF_COMPAT_PATCHED = False


def patch_urdf_importer_if_needed() -> bool:
    """Patch Isaac Lab's URDF converter when only importer 2.4.30 is present.

    Returns:
        ``True`` when the fallback path was installed, ``False`` when the
        native pinned importer is available or when Isaac Sim is not running.

    The function deliberately catches import/runtime failures.  Importing the
    Python package in a unit-test process without Kit must remain harmless; an
    actual Isaac Sim process will surface conversion errors when the asset is
    spawned if its importer is genuinely unusable.
    """

    global _URDF_COMPAT_PATCHED
    if _URDF_COMPAT_PATCHED:
        return True

    try:
        import omni.kit.app
        from isaaclab.sim.converters import urdf_converter
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
    except (ImportError, ModuleNotFoundError):
        return False

    try:
        manager = omni.kit.app.get_app().get_extension_manager()
    except Exception:
        return False
    pinned_name = "isaacsim.asset.importer.urdf-2.4.31"
    fallback_name = "isaacsim.asset.importer.urdf"

    # A matched Isaac Lab/Sim environment should use the version-pinned path.
    # ``get_enabled_extension_id`` is not a reliable availability probe on
    # Kit 107: for an unresolved version it can still return the unversioned
    # extension id.  The explicit enabled-state query distinguishes the two
    # cases and keeps the normal matched path untouched.
    try:
        if manager.is_extension_enabled(pinned_name):
            return False
    except Exception:
        pass

    # Enable the unversioned importer shipped by the current Isaac Sim pip
    # environment.  Kit may already have loaded it, so this operation is
    # intentionally idempotent.
    try:
        manager.set_extension_enabled_immediate(fallback_name, True)
    except Exception:
        return False

    # If the unversioned extension itself is the 2.4.31 build, use it without
    # installing any shim.  The manifest is local and avoids asking Kit's
    # extension registry (which is often unavailable on training machines).
    try:
        extension_id = manager.get_enabled_extension_id(fallback_name)
        extension_root = Path(manager.get_extension_path(extension_id))
        manifest_candidates = (extension_root / "config" / "extension.toml", extension_root / "extension.toml")
        manifest_text = next((path.read_text() for path in manifest_candidates if path.is_file()), "")
        if re.search(r"^version\s*=\s*[\"']2\.4\.31(?:[\"']|\+)", manifest_text, re.MULTILINE):
            return False
    except Exception:
        # Missing metadata is not fatal; the ImportConfig feature probe below
        # remains the authoritative compatibility check.
        pass

    # Isaac Sim 5.1's 2.4.30 ImportConfig lacks the setter added in 2.4.31.
    # Add an instance-method no-op only when the method is actually absent.
    try:
        import omni.kit.commands

        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config_type = type(import_config)
        if not hasattr(import_config_type, "set_merge_fixed_ignore_inertia"):
            # PyBind-generated classes differ between Isaac Sim patch releases:
            # some allow instance attributes, others only class attributes.
            # Try both forms before giving up and leaving the native converter
            # untouched.
            try:
                setattr(import_config, "set_merge_fixed_ignore_inertia", lambda value: None)
            except Exception:
                setattr(import_config_type, "set_merge_fixed_ignore_inertia", lambda self, value: None)
    except Exception:
        return False

    # Avoid the converter's unconditional request for the unavailable 2.4.31
    # extension, while retaining AssetConverterBase's hashing/lazy conversion
    # behavior and the normal _get_urdf_import_config implementation above.
    original_init = urdf_converter.UrdfConverter.__init__
    if getattr(original_init, "_dribblebot_urdf_compat", False):
        _URDF_COMPAT_PATCHED = True
        return True

    def _compat_init(self: Any, cfg: Any) -> None:
        from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface

        self._urdf_interface = acquire_urdf_interface()
        AssetConverterBase.__init__(self, cfg=cfg)

    _compat_init._dribblebot_urdf_compat = True  # type: ignore[attr-defined]
    urdf_converter.UrdfConverter.__init__ = _compat_init
    _URDF_COMPAT_PATCHED = True
    return True
