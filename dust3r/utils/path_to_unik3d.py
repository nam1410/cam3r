# Copyright (C) 2026-present Namitha Guruprasad and the CAM3R authors.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Make the vendored UniK3D package importable as `unik3d`.
#
# The package lives at <repo>/UniK3D/unik3d, so `import unik3d` needs
# <repo>/UniK3D on sys.path. This mirrors path_to_croco.py, which does the
# same for the vendored croco checkout.
# --------------------------------------------------------
import sys
import os.path as path

HERE_PATH = path.normpath(path.dirname(__file__))
UNIK3D_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../UniK3D'))
UNIK3D_PACKAGE_PATH = path.join(UNIK3D_REPO_PATH, 'unik3d')

if path.isdir(UNIK3D_PACKAGE_PATH):
    if UNIK3D_REPO_PATH not in sys.path:
        sys.path.insert(0, UNIK3D_REPO_PATH)
else:
    raise ImportError(
        f"UniK3D is not available, could not find: {UNIK3D_PACKAGE_PATH}.\n"
        "The UniK3D/ directory ships with this repository; if it is missing, re-clone it."
    )
