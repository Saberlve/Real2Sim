# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Set hardcoded-macros
# TODO: Make these configurable and download the checkpoints at the start
ROOT_DIR = os.path.dirname(__file__)
REPO_DIR = '/'.join(ROOT_DIR.split('/')[:-1])
CHECKPOINT_DIR = f"{REPO_DIR}/checkpoints"
ASSET_DIR = f"{REPO_DIR}/assets" # TODO: maybe a way to detect this?
CFG_DIR = f"{REPO_DIR}/scripts/cfg"
DATA_DIR = f"{REPO_DIR}/Data"

# Prefer the shared NAS1 checkpoints when available. Individual model wrappers
# use these variables and fail clearly if the requested local checkpoint is absent.
os.environ.setdefault(
    "SIMFOUNDRY_SAM3_CHECKPOINT",
    "/run/determined/NAS1/public/HuggingFace/facebook/sam3/sam3.pt",
)
os.environ.setdefault(
    "SIMFOUNDRY_DINOV3_MODEL",
    "/run/determined/NAS1/public/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
)


def get_omnigibson_data_path() -> str:
    """Return the repo-local OmniGibson dataset root used by pipeline assets."""
    return os.path.join(REPO_DIR, "deps", "BEHAVIOR-1K", "datasets")


def configure_omnigibson_data_path(*, force: bool = True) -> str:
    """Point OmniGibson at this checkout's dataset root before importing it."""
    dataset_root = get_omnigibson_data_path()
    if os.path.isdir(dataset_root) and (force or not os.environ.get("OMNIGIBSON_DATA_PATH")):
        os.environ["OMNIGIBSON_DATA_PATH"] = dataset_root
    return os.environ.get("OMNIGIBSON_DATA_PATH", dataset_root)


# For importing specific OmniGibson dependences
def import_og_dependencies():
    from simfoundry.tasks.pick_place_task import PickPlaceTask
