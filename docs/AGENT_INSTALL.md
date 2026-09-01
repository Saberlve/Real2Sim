# SimFoundry — Agent Installation Guide

A procedural guide for a coding agent (Claude Code, Codex, etc.) installing SimFoundry
on a fresh Linux + NVIDIA machine.

**Read this whole file before running anything.** The install takes hours and several
steps are effectively irreversible once started.



## 0. Ground rules for agents

- **You cannot complete OAuth flows.** `gcloud auth application-default login` and
  `hf auth login` (interactive) open a browser. Use the non-interactive paths in §3.
  If credentials are absent and you cannot obtain them, **stop and ask the user** —
  do not attempt to work around auth.
- **Never delete a conda env without confirming it with the user first.** Env names
  are not reliably namespaced; unrelated projects live alongside SimFoundry.
- **Run the installer detached with a logfile.** It runs for hours; a dropped
  foreground process loses everything.
- **`install_everything.sh` skips envs that already exist.** A half-built env from a
  crashed run is silently treated as done. After any failure, delete the specific
  broken env before re-running (see §6).

---

## 1. Preflight

Run all of these and confirm before touching anything.

```bash
# GPU + driver
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Toolchain — all must resolve
command -v mamba git-lfs ffmpeg gcloud

# CUDA 12.8 must exist at this exact path (install_any6d.sh hard-codes it)
ls -d /usr/local/cuda-12.8

# Disk: need ~250 GB free for all 7 envs + deps + checkpoints
df -h .
```

**Hard requirements:**

| Requirement | Why | Failure if missing |
|---|---|---|
| `mamba` (Miniforge) | every installer calls it | exits 127 immediately |
| `/usr/local/cuda-12.8` | `install_any6d.sh` sets `CUDA_HOME` to it | any6d build fails |
| `git-lfs` | articulation repos store assets in LFS | corrupt checkouts |
| ~250 GB free disk | envs ≈ 100 GB, deps ≈ 82 GB, HF cache ≈ 12 GB | mid-build ENOSPC |
| ≥ 16 GiB VRAM | stage 5 peaks ~14 GiB; budget is 90% of total | stage 5 rejected pre-flight |

The repo has no git submodules. All dependencies are
cloned by the install scripts into `deps/`.
---

## 2. Choose the install scope

> Fresh clones of `deps/` repos are checked out to their pinned commits automatically —
> the guard in `git_safe.sh` compares a branch against *its own remote*, so only a dirty
> tree or genuinely unpushed local commits skip the pin (with a loud `NOTE:`). Set
> `SIMFOUNDRY_FORCE_DEP_CHECKOUT=1` only to overwrite such local state deliberately.
> See §6.1.

```bash
# All 7 envs (simfoundry, hunyuan, any6d, da3, void, nerfstudio_simfoundry, 3dgrut)
bash scripts/installation/install_everything.sh

# Core reconstruction only — skips the auto-background trio
bash scripts/installation/install_everything.sh --only "simfoundry hunyuan any6d da3"
```

The auto-background envs (`void`, `nerfstudio_simfoundry`, `3dgrut`) are only needed
for the Gaussian-splat background flow, which is **opt-in** (`--bg-splat`). Skip them
unless the user asked for backgrounds.

Always run detached with a log:

```bash
mkdir -p ~/simfoundry_logs
bash scripts/installation/install_everything.sh > ~/simfoundry_logs/install.log 2>&1
```

Then watch for phase transitions and failures:

```bash
tail -f ~/simfoundry_logs/install.log \
  | grep -E --line-buffered ">>> \[|DONE|Error occurred at line|^Error:|Traceback|No space left|Killed"
```

Each env begins with a `>>> [name] install_X.sh (env: name)` line. That is your
progress marker.

---

## 3. Authentication (do this before checkpoints)

Checkpoints are **opt-in** and require Hugging Face auth, so the order is:
install envs → authenticate → download checkpoints.

### 3a. Hugging Face


The repo docs demonstrate login, `hf auth login`.

Check existing auth first — it usually already exists and needs nothing:

```bash
hf auth whoami
```

If that prints a username, you are done. Otherwise the user must supply a token
(`HF_TOKEN`). Write it into the keys file without echoing it:

```bash
cd scripts/installation
cp api_keys.template.txt api_keys.txt
# then edit api_keys.txt — set HF_TOKEN and GCLOUD_PROJECT
chmod 600 api_keys.txt
```

**Gated models.** The user's HF account must have approved access to these, or later
stages fail with a 401/403 that does not name the cause:

```bash
for r in facebook/sam3 \
         facebook/dinov3-vitl16-pretrain-lvd1689m \
         briaai/RMBG-2.0 \
         netflix/void-model; do
  printf '%-50s ' "$r"
  hf download "$r" config.json --quiet >/dev/null 2>&1 && echo OK || echo DENIED
done
```

`black-forest-labs/FLUX.1-Kontext-dev` is optional — only needed if a config sets
`model: flux`. Legacy configs may use Gemini; the current reconstruction defaults use
Qwen as described below.

### 3b. Google Cloud / Gemini (optional when using Qwen)

Only stages/configurations that select a Gemini model need Gemini credentials. Qwen is
the low-cost default for the reconstruction VLM/image stages described in §3d. If a
configuration still selects Gemini, there are two authentication routes:

**Vertex AI (default).** Needs Application Default Credentials, which require an
interactive browser flow an agent cannot perform:

```bash
gcloud auth application-default login   # INTERACTIVE — user must run this
export GCLOUD_PROJECT=<project-id>
```

Verify without triggering the flow:

```bash
ls ~/.config/gcloud/application_default_credentials.json && gcloud config get-value project
```

**API key (agent-friendly fallback).** Generate a key at
<https://aistudio.google.com/api-keys>, then:

```bash
export GEMINI_API_KEY=<key>
```

`simfoundry/models/vlm.py::resolve_gemini_auth` prefers an API key when present and
falls back to Vertex+ADC otherwise. A file named `api_keys.txt` in the repo root (or
any parent dir) is auto-loaded into the environment by `load_api_keys()`, so
`GEMINI_API_KEY=...` in `<repo>/api_keys.txt` works without exporting anything.

### 3c. DeepSeek V4 Flash Vision

The VLM adapter also supports DeepSeek's OpenAI-compatible vision model
`deepseek-v4-flash-vision-exp`. Add the key to the repo-root `api_keys.txt` (or export
it) and select that model in the relevant YAML stage:

```text
DEEPSEEK_API_KEY=<key>
```

For example, set `s3_ground.detection_model` and/or `s5_scene.detection_model` to
`deepseek-v4-flash-vision-exp`. The default endpoint is `https://api.deepseek.com`;
set `DEEPSEEK_BASE_URL` only when using a compatible proxy. `gcloud_project` and
Gemini credentials are not needed by those DeepSeek calls. This model is experimental,
so its availability and API behavior may change.

### 3d. Qwen vision (low-cost default)

The default reconstruction configuration uses `qwen3-vl-flash` for vision understanding
and `qwen-image-2.0` for image removal/upsampling in stages 5 and 6. Export the DashScope
key before running:

```bash
export DASHSCOPE_API_KEY=<key>
```

The adapter uses the OpenAI-compatible DashScope endpoint
`https://dashscope.aliyuncs.com/compatible-mode/v1` for Qwen-VL. Override it with
`DASHSCOPE_BASE_URL` when using a workspace or another region. Qwen Image editing uses
the native endpoint; override it with `DASHSCOPE_IMAGE_API_URL` if needed. The adapter
downloads the temporary output URL immediately, as DashScope output URLs expire.

### 3e. Local NAS1 vision checkpoints

When `/run/determined/NAS1` is mounted, SimFoundry uses the local NAS1 SAM3 checkpoint
instead of downloading `facebook/sam3`:

```bash
export SIMFOUNDRY_SAM3_CHECKPOINT=/run/determined/NAS1/public/HuggingFace/facebook/sam3/sam3.pt
export SIMFOUNDRY_DINOV3_MODEL=/run/determined/NAS1/public/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

The current A reconstruction code does not instantiate a DINOv3 encoder directly;
the DINOv3 path is exposed for components that do. Override either variable when
running on a machine with a different NAS mount.

Note this root `api_keys.txt` is a *different file* from
`scripts/installation/api_keys.txt` used by `login_services.sh`.

### 3f. Non-interactive login

```bash
bash scripts/installation/login_services.sh --default
```

Reads `scripts/installation/api_keys.txt`. Blank values are skipped, so a file with
only `HF_TOKEN` and `GCLOUD_PROJECT` is fine.

---

## 4. Checkpoints

```bash
bash scripts/installation/download_checkpoints.sh --default
```

**This is the flakiest step.** FoundationStereo and FoundationPose weights come from
Google Drive via `gdown --folder`, which is rate-limited and fails intermittently with
no useful error. If it fails, re-run it — it is idempotent and skips existing files.

If a machine already has a good copy:

```bash
bash scripts/installation/download_checkpoints.sh --default \
  --checkpoint-fallback-root /path/to/known-good/repo-copy
```

Weights land in `deps/void-model/` (VOID, ~41 GB) and `checkpoints/`. Do not move
them — several runners resolve paths against `VOID_ROOT=deps/void-model`.

---

## 5. Verification

```bash
# Each must print a path inside THIS checkout
for e in simfoundry any6d da3 hunyuan; do
  printf '%-12s ' "$e"
  mamba run -n "$e" python -c "import simfoundry; print(simfoundry.__file__)" 2>&1 | tail -1
done

# Catches both §6.1 failures at once. Expect: lerobot 0.3.4, numpy 1.26.4,
# torch 2.x+cu128, cuda True, and a simfoundry path inside this checkout.
mamba run -n simfoundry python -c "
import lerobot, omnigibson, simfoundry, torch, numpy
print('lerobot   ', lerobot.__version__)
print('numpy     ', numpy.__version__)
print('torch     ', torch.__version__, 'cuda', torch.cuda.is_available())
print('simfoundry', simfoundry.__file__)"

# Stage-7 and VLM-specific runtime checks
mamba run -n hunyuan python -c \
  "import bpy, diffusers, pytorch_lightning, trimesh, xatlas; import custom_rasterizer as cr; assert callable(cr.rasterize)"
mamba run -n simfoundry python -c \
  "import flash_attn; assert flash_attn.__version__ == '2.7.4.post1'"

# Stage plans (executes nothing)
bash scripts/pipeline/A_reconstruction/run.sh --dry-run --include 1b,2
bash scripts/pipeline/B_augmentation/run.sh   --dry-run --include 1
bash scripts/pipeline/C_application/run.sh    --dry-run --mode smoke-random

# Tests
mamba run -n simfoundry python -m pytest -q
```

A path outside the checkout means a stale editable install is shadowing the package —
`pip uninstall simfoundry` in that env and re-run its installer.

Note: the four-file test subset in `docs/INSTALL.md` runs before any environment is
built, but needs `pytest` in whatever Python you invoke — it is not in a stock
Miniforge `base` (`pip install pytest` first, as the docs say).

---

## 6. Known failure modes



### `ERROR: Required OmniGibson robot asset is missing: .../franka_robotiq.usda`

The `franka_robotiq` end effector download failed. It comes from the **public Hugging
Face dataset** `behavior-1k/omnigibson-robot-assets` (a ~225 MB subtree via
`snapshot_download`, no token needed), fetched after OmniGibson's own public asset
download — which does *not* carry `franka_robotiq` on its own.

Causes:

- **HF unreachable / download failed.** Re-run the installer (the fetch is idempotent
  and skips when `franka_robotiq.usda` already exists), or point
  `OG_ROBOT_ASSETS_HF_REPO` at a mirror with the same dataset layout.
- **Fully offline machine with a local copy.** `--robot-asset-fallback-root` expects the
  layout `<root>/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets/<rel_path>`; a copy
  in any other layout needs a symlink shim:

```bash
SHIM=/tmp/asset_fallback
mkdir -p "$SHIM/deps/BEHAVIOR-1K/datasets"
ln -sfn /path/to/omnigibson-robot-assets "$SHIM/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets"

bash scripts/installation/install_simfoundry.sh \
  --project-root "$PWD" --env-name simfoundry --default \
  --robot-asset-fallback-root "$SHIM"
```

**`install_everything.sh` does not accept or forward `--robot-asset-fallback-root`**, so
in the fallback case the `simfoundry` env must be built by calling
`install_simfoundry.sh` directly. Afterwards, re-run `install_everything.sh` normally —
it skips the existing `simfoundry` env and continues with the other six.

### Installer stops partway
`install_everything.sh` uses `set -euo pipefail`, so the first failure aborts every
remaining env. Because re-runs **skip existing envs**, a half-built env is treated as
complete. Always delete the broken env explicitly before re-running:

```bash
mamba env remove -n <broken-env> -y
rm -rf ~/miniforge3/envs/<broken-env>     # mamba sometimes leaves a stub
bash scripts/installation/install_everything.sh          # resumes at that env
```

### `ImportError: cannot import name 'BaseRobot' from 'omnigibson.robots'`
`deps/BEHAVIOR-1K` is on `main` instead of the pinned commit. OmniGibson main renamed
`BaseRobot`→`Robot` and dropped `FrankaPanda`. Fix:

```bash
grep -n 'BEHAVIOR1K_COMMIT=' scripts/installation/install_simfoundry.sh
git -C deps/BEHAVIOR-1K rev-parse HEAD    # must match
```

### `ImportError: cannot import name 'HF_LEROBOT_HOME'`
lerobot is too new. The pinned OmniGibson 3.8.0 needs lerobot 0.3.4:

```bash
mamba run -n simfoundry pip install --no-deps \
  "lerobot@git+https://github.com/huggingface/lerobot.git@577cd10974b84bea1f06b6472eb9e5e74e07f77a"
mamba run -n simfoundry python -c "import numpy; print(numpy.__version__)"   # expect 1.26.4
```

### Stage 2c fails with `ns-process-data: not found`
Stage 2c runs in `nerfstudio_simfoundry`, not `simfoundry`. It is opt-in — omit
`--bg-splat` if you did not build that env.

### Stage 7 Hunyuan missing packages or `custom_rasterizer` import errors

Stage 7 runs in the separate `hunyuan` environment. The installer must install the
runtime packages explicitly, even though most are listed by upstream Hunyuan3D:
`bpy==4.0.0`, `diffusers==0.30.0`, `transformers==4.46.0`,
`pytorch-lightning==1.9.5`, `realesrgan==0.3.0`, `basicsr==1.4.2`,
`fast-simplification==0.2.0`, `pymeshlab==2022.2.post4`, `xatlas==0.0.9`, and
`trimesh==4.5.1`. DeepSpeed is intentionally excluded from the runtime install: it
is not imported by the reconstruction path and its metadata build can abort the whole
requirements transaction.

The installer also writes
`$CONDA_PREFIX/etc/conda/activate.d/simfoundry_hunyuan_runtime.sh`, which exposes the
in-place rasterizer package and Torch's shared libraries. After activating `hunyuan`,
verify it before running stage 7:

```bash
mamba activate hunyuan
python -c 'import custom_rasterizer as cr; assert callable(cr.rasterize); print("custom_rasterizer OK")'
```

If the environment predates this installer change, repair the current shell manually:

```bash
export PYTHONPATH="$PWD/deps/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

### Stage 7 VRAM / Hunyuan `low_vram` device mismatch

The current Hunyuan integration's `enable_model_cpu_offload()` path can fail with
`Expected all tensors to be on the same device, cuda:0 and cpu`. Therefore do not rely
on `s7_mesh.low_vram=true` as the generic fix. Use `low_vram=false` when the card can
hold the shape model, and reduce peak memory by running shape and texture as two
separate invocations:

```bash
export PYTHONPATH="$PWD/deps/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Shape pass
mamba run -n hunyuan python scripts/pipeline/A_reconstruction/stages/7_generate_object_meshes.py \
  root_dir=/path/to/SimFoundry/Data scene_name=<name> \
  s1_video.video_fpath=/path/to/video.mp4 \
  s7_mesh.low_vram=false s7_mesh.generate_shape=true s7_mesh.generate_texture=false

# Texture pass; reuses the shape artifacts
mamba run -n hunyuan python scripts/pipeline/A_reconstruction/stages/7_generate_object_meshes.py \
  root_dir=/path/to/SimFoundry/Data scene_name=<name> \
  s1_video.video_fpath=/path/to/video.mp4 \
  s7_mesh.low_vram=false s7_mesh.generate_shape=false s7_mesh.generate_texture=true
```

On a card that cannot hold the shape model even with the split, reduce
`s7_mesh.object_indices` and process objects in smaller batches. `--no-stream` is a
separate scheduling option and does not repair the device mismatch.

---

## 7. Running the pipeline

These commands describe the standard end-to-end flow. For Hunyuan stage 7, use the
split shape/texture procedure in §6 when `low_vram=true` triggers the device mismatch.

```bash
# Only needed for stages/configs that select Gemini or Vertex-backed services.
export GCLOUD_PROJECT=<project>

# A — reconstruction (~20 min for a 3-object tabletop scene)
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name <name> --video-fpath /path/to/video.mov

# B — augmentation
bash scripts/pipeline/B_augmentation/run.sh --scene-name <name>

# C — OmniGibson smoke test
bash scripts/pipeline/C_application/run.sh --scene-name <name> --mode smoke-random
```

For Hunyuan stage 7, `s7_mesh.low_vram=false` is the reliable setting for the current
integration. On smaller cards, run the shape and texture passes separately as shown in
§6 and optionally restrict `s7_mesh.object_indices`.

**Expect some benign noise in headless logs**, none of it fatal:
- `AttributeError: 'NoneType' object has no attribute 'GetCamera'` from
  `omni.kit.widget.viewport` — Isaac Sim's headless shutdown, fires repeatedly in stages
  11-14.

The installer pins SimFoundry's Flash-Attention to `2.7.4.post1`, which satisfies the
diffusers/Flux compatibility check (`>=2.7.1` and the 2.7.4 line). If a pre-existing
environment still reports `got 2.8.3`, reinstall it with:

```bash
mamba run -n simfoundry python -m pip install --no-build-isolation \
  "flash-attn==2.7.4.post1"
```

Filter both out when monitoring, or real failures get lost in them.

Useful flags: `--include`/`--exclude` to select stages, `--dry-run` to print the plan,
`--detect-articulation` for stage 9, `--bg-splat` for stage 2c.

**Articulation needs a minimum of 18 GiB VRAM.** The 16 GiB minimum covers only the
standard pipeline; stage 9's segmentation models allocate outside the VRAM scheduler.

**`--detect-articulation` degrades gracefully.** The availability check now verifies the
`deps/articulate-anything` checkout and the `articulate-anything-*` conda envs, not just
the stage script; if they are missing, the flag is ignored with a warning and the rest
of the pipeline runs. To confirm articulation will actually run:

```bash
mamba env list | grep articulate-anything
```

Env-name overrides, if yours differ from the defaults:
`--env-simfoundry`, `--env-da3`, `--env-mesh`, `--env-nerfstudio`, `--env-b1k`.
`--env-mesh` must match the backend in `s7_mesh.shape_model` (`hunyuan` → `hunyuan` env).

---

## 8. Quick reference

| Env | Built by | Used for |
|---|---|---|
| `simfoundry` | `install_simfoundry.sh` | most stages, VLM calls, OmniGibson |
| `hunyuan` | `install_hunyuan.sh` | stage 7 / B stage 3 mesh generation |
| `any6d` | `install_any6d.sh` | stage 8 pose matching |
| `da3` | `install_da3.sh` | stage 2 depth |
| `void` | `install_void.sh` | auto-BG inpainting (optional) |
| `nerfstudio_simfoundry` | `install_nerfstudio.sh` | stage 2c splat training (optional) |
| `3dgrut` | `install_3dgrut.sh` | PLY → USDZ (optional) |
| `articulate-anything-{hunyuan,partfield}` | `install_articulate.sh` | stage 9 (optional) |

There is **no** `b1k` env, and none is needed — `--env-b1k` defaults to `simfoundry`
in all three pipelines. Pass it only if you keep OmniGibson in a separate environment.

---

## 9. Installation incident log (2026-08)

This section records the problems encountered during the last full installation. Keep
these rules when repeating the installation; they prevent the same failures from being
reintroduced by a future agent or by a fresh checkout.

### 9.1 Conda-only policy

The installation was initially mixed between `uv` and Conda. This is unsafe here:
`uv` can create an environment that is different from the environment selected by the
installer, and compiled CUDA packages then land in the wrong prefix. The project was
standardized on the existing Conda environments:

- Do not create a `.venv` or install an environment with `uv`.
- Use `mamba create`, `mamba install`, and the selected environment's
  `python -m pip` only.
- `install_3dgrut.sh` and the vendored 3DGRUT install helpers were adjusted to avoid
  installing or invoking `uv`; `INSTALL_TCNN_WITH_UV=0` is used for that installer.
- Always validate compiled modules with `mamba run -n <env> ...` or after activating
  the environment. Calling an environment's Python by an absolute path can omit its
  CUDA library activation and produce misleading `libc10.so` errors.

### 9.2 Proxy, credentials, and cloning

All network operations must inherit the proxy and credentials configured in the user's
shell startup file:

```bash
source ~/.bashrc
git clone ...
hf download ...
python -m pip install ...
```

The HF token was already provided as `HF_TOKEN` in `~/.bashrc`; do not print it, put it
in a log, or start an interactive `hf auth login`. Google Cloud login was deliberately
not performed. SAM and DINOv3 were already available on NAS1, so they must not be
downloaded again unless the NAS paths are absent.

If a clone or model download fails with a network error, first confirm that the command
was run after `source ~/.bashrc`; do not replace the configured proxy with a hard-coded
proxy value.

### 9.3 CUDA toolkit solver failures

Several installers originally requested `cuda-toolkit` from only the NVIDIA channel.
Mamba could not solve or locate the requested package in that configuration. The CUDA
toolkit installs now specify compatible channels explicitly:

```bash
mamba install -y -c nvidia -c conda-forge -c defaults cuda-toolkit=12.8
```

The same channel policy is used by Hunyuan3D, DA3, Any6D, and Nerfstudio. A local
`simfoundry-condarc` keeps the channel order reproducible. Do not let 3DGRUT silently
select CUDA 12.9: its environment must use CUDA 12.8 to match the PyTorch `cu128`
build and the compiled extensions.

### 9.4 Hunyuan3D dependency installation aborted by DeepSpeed

Hunyuan3D's unfiltered requirements install attempted to build/install DeepSpeed before
the environment's CUDA compiler variables were ready. Its metadata step failed and
aborted the entire requirements transaction, leaving ordinary packages such as
`trimesh`, `diffusers`, and `transformers` missing. The repair was:

1. Install the environment CUDA toolkit and export `CUDA_HOME`, include paths, and
   library paths first.
2. Install requirements while excluding `deepspeed` and third-party mirror index
   options.
3. Install the required `trimesh` version explicitly, then finish the Hunyuan and
   SimFoundry editable installs.

Do not interpret a failed DeepSpeed build as evidence that the whole Hunyuan
environment is usable.

### 9.5 FAISS GPU package mismatch

DA3 uses Python 3.11, but the Conda `faiss-gpu=1.12` build available from the selected
channels was only for Python 3.10. Mamba therefore could not solve FAISS for DA3. The
working Python 3.11 CUDA 12 wheel is:

```bash
mamba run -n da3 python -m pip install faiss-gpu-cu12==1.12.0
```

Any6D uses Python 3.10, so its compatible Conda package can be installed normally:

```bash
mamba install -n any6d --override-channels \
  -c pytorch -c conda-forge faiss-gpu=1.12 -y
```

Verify both with `faiss.get_num_gpus()` and ensure it reports the visible GPUs. Do not
blindly use the Python 3.10 Conda build in DA3.

### 9.6 Any6D CUDA extension import error

`common` and `gridencoder` were successfully compiled, but importing them through an
unactivated absolute-path interpreter reported `libc10.so: cannot open shared object
file`. This was an environment activation/library-path issue, not a failed build.
Validate as follows:

```bash
source ~/.bashrc
mamba run -n any6d python -c \
  "import torch, common, gridencoder; print(torch.cuda.is_available())"
```

The final Any6D check must also import `sam2`, `bop_toolkit_lib`, and `simfoundry`.

### 9.7 3DGRUT build and Kaolin imports

The original 3DGRUT helper installed `uv` and selected a CUDA 12.9 toolkit. This caused
toolchain inconsistency and had to be replaced with Conda's Python/pip workflow and
CUDA 12.8. After the native extensions built, Kaolin still failed at import time due
to missing runtime Python dependencies. Installing the following resolved the import:

```bash
mamba run -n 3dgrut python -m pip install \
  pygltflib comm flask ipycanvas ipyevents pybind11 warp-lang 'jupyter_client<8'
```

The final check must import `threedgrut`, `tinycudann`, `kaolin`, `ppisp`, and
`fused_ssim` from `mamba run -n 3dgrut`.

### 9.8 Checkpoint script argument and download completion

The requested command with a trailing repository argument failed because the script
does not accept positional arguments:

```text
Unknown option: .
```

The correct command is:

```bash
source ~/.bashrc
bash scripts/installation/download_checkpoints.sh --default
```

The script is idempotent. Re-running it is safe when a Google Drive or Hugging Face
download is interrupted. Always wait for `All checkpoints accounted for.` and verify
the files, rather than relying only on the process exit status. RMBG-2.0 belongs in
`$DATA_HOME/RMBG-2.0`; the existing NAS1 SAM/DINOv3 assets should remain referenced in
place.

### 9.9 Final repeatable audit

Run this after any repair or resumed installation. It catches the key failures above
without importing through the wrong Python prefix:

```bash
source ~/.bashrc
for env in simfoundry da3 hunyuan any6d void nerfstudio_simfoundry 3dgrut; do
  mamba run -n "$env" python -c \
    "import torch; print('$env', torch.__version__, torch.cuda.is_available())" || exit 1
done

mamba run -n da3 python -c \
  "import faiss; print(faiss.__version__, faiss.get_num_gpus())"
mamba run -n any6d python -c \
  "import common, gridencoder, sam2, bop_toolkit_lib"
mamba run -n 3dgrut python -c \
  "import threedgrut, tinycudann, kaolin, ppisp, fused_ssim"
test -s "$DATA_HOME/RMBG-2.0/model.safetensors"
bash scripts/installation/download_checkpoints.sh --default
```

### 9.10 Hunyuan stage-7 runtime and CUDA library paths

During the first stage-7 attempt, `trimesh`, `cv2`, and then
`pytorch_lightning` were missing because the upstream Hunyuan requirements transaction
was interrupted by DeepSpeed. The installer now excludes DeepSpeed, explicitly installs
the complete runtime set, and re-pins the NumPy-2-compatible packages after the
SimFoundry editable install. The final OpenCV package is
`opencv-python-headless==4.11.0.86`; do not install a second `opencv-python` variant
afterwards.

The compiled `custom_rasterizer_kernel` links against `libc10.so` and related Torch
libraries under `site-packages/torch/lib`. Merely activating the conda environment is
not sufficient on every machine, so `install_hunyuan.sh` persists both
`LD_LIBRARY_PATH` and `PYTHONPATH` in the Hunyuan activation hook and runs an import
smoke test before finishing.

### 9.11 Flash-Attention and optional FLUX imports

The VLM module imports diffusers' Flux Kontext pipeline while loading the VLM adapter.
That import checks Flash-Attention even when the selected model is Qwen. Version 2.8.3
failed with:

```text
Requires Flash-Attention version >=2.7.1,<=2.7.4 but got 2.8.3
```

`install_simfoundry.sh` now installs and verifies `flash-attn==2.7.4.post1`. Do not
replace it with the previously commented 2.8.3 wheel in the installer.

### 9.12 NAS1 model cache and local vision assets

When NAS1 is mounted, the SimFoundry activation hook automatically uses these files if
they exist:

```text
/run/determined/NAS1/public/HuggingFace/facebook/sam3/sam3.pt
/run/determined/NAS1/public/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
/run/determined/NAS1/public/HuggingFace/vla/hub/
```

It sets `SIMFOUNDRY_SAM3_CHECKPOINT`, `SIMFOUNDRY_DINOV3_MODEL`, `HF_HOME`, and
`HF_HUB_CACHE` only when the corresponding variables were not already provided by the
user. This prevents a fresh install from downloading SAM3, DINOv3, or Prior Depth
Anything again when the local cache is available.

### 9.13 Headless Omniverse and USD import

Stages 10-13 can run without an X server. The installer persists
`OMNI_KIT_ACCEPT_EULA=YES` and `OMNIGIBSON_HEADLESS=1` in the SimFoundry activation
hook. For a completely display-less shell, also prepare a private runtime directory:

```bash
export DISPLAY=
export XDG_RUNTIME_DIR=/tmp/simfoundry-runtime
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
```

GLFW, audio, NGX, and viewport-camera warnings can still appear during Isaac Sim
startup or teardown. Treat them as warnings if the stage writes its success marker and
USD output; do not install a GUI desktop stack just to silence those messages.

### 9.14 Optional Newton cross-project smoke test

The SimFoundry installer does not own the tactile-benchmark environment. After the USD
stages succeed, its local Newton check should be run from the sibling project with that
project's Python and source path:

```bash
cd ../tactile-benchmark
PYTHONPATH=. .venv/bin/python -c '
from tacsim.newton_runtime import activate_newton
activate_newton()
import newton.usd
print("Newton USD import OK")
'
```

Do not use the system Python for this check and do not install a second Newton copy into
SimFoundry. The benchmark's `third_party/newton` checkout and `usd-core` belong to its
`.venv`.

### 9.15 Reconstruction stage names and USD artifacts

There is no `9_generate_scene.py` in the current checkout. The final A stages are:

```text
9_compile_scene.py
10_make_objects_sim_ready.py
11_stabilize_physics.py
12_import_usd.py
13_create_og_scene.py
```

Stage 12 imports the object USD files into the BEHAVIOR-1K asset tree; stage 13 creates
the OmniGibson scene wrapper. A successful stage-10 log alone is not the final USD
check. Check `s12_usd/stage_info.json`, then inspect the generated object USD under
`deps/BEHAVIOR-1K/datasets/real2sim-assets/objects/`.
