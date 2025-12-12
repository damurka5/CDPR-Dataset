# CDPR Environment Setup Guide

**macOS / Linux / Windows**

This guide explains how to set up a local environment to run the **CDPR MuJoCo simulator**, generate datasets, and collect **teleoperated demonstrations**.
It does **not** require installing OpenVLA or any vision-language models.

---

## 0. Overview

This setup supports:

* MuJoCo-based CDPR simulation
* Scene wrapper generation
* Synthetic dataset generation
* Keyboard teleoperation + trajectory recording
* RLDS-compatible outputs

### What is NOT required locally

* OpenVLA / OpenVLA-OFT
* CUDA
* GPU drivers (CPU rendering is sufficient)

---

## 1. Repository layout (expected)

All repositories should live under a common directory, e.g.:

```text
<workspace>/
├── VLA_CDPR/
│   └── cdpr_mujoco/
├── CDPR-Dataset/
│   └── cdpr_dataset/
```

Example:

```text
~/Projects/CDPR/
├── VLA_CDPR
├── CDPR-Dataset
```

---

## 2. Conda environment (all OS)

### 2.1 Install Conda

* macOS / Linux: Miniconda or Mambaforge
* Windows: Miniconda (Anaconda Prompt recommended)

### 2.2 Create environment

```bash
conda create -n cdpr python=3.10 -y
conda activate cdpr
```

---

## 3. Core Python dependencies

Install common scientific + IO packages:

```bash
conda install -c conda-forge \
    numpy scipy matplotlib \
    imageio imageio-ffmpeg \
    pyyaml \
    opencv \
    pip \
    -y
```

---

## 4. MuJoCo installation

### 4.1 Install MuJoCo Python bindings

```bash
pip install mujoco==3.3.7 glfw PyOpenGL
```

> MuJoCo 3.x includes binaries — no license or manual download required.

---

## 5. Install CDPR MuJoCo code (VLA_CDPR)

From your workspace root:

```bash
pip install -e ./VLA_CDPR/cdpr_mujoco
```

Verify:

```bash
python - <<EOF
import mujoco
import cdpr_mujoco
print("MuJoCo + cdpr_mujoco OK")
EOF
```

---

## 6. Install CDPR-Dataset

### Option A (recommended): editable install

Ensure `CDPR-Dataset/pyproject.toml` exists:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cdpr-dataset"
version = "0.0.0"
requires-python = ">=3.10"

[tool.setuptools]
packages = ["cdpr_dataset"]
```

Then:

```bash
pip install -e ./CDPR-Dataset
```

### Option B: run in-place (no install)

Always run commands **from inside** `CDPR-Dataset/`:

```bash
cd CDPR-Dataset
python -m cdpr_dataset.generate_cdpr_dataset --help
```

You’re right — if your CDPR scene switcher / assets rely on LIBERO objects (milk, plate, etc.), then **LIBERO (and its assets) must be installed locally** too. Here’s the missing section you can paste into the guide.

---

## Add to the Markdown guide: LIBERO installation

### 6. Install LIBERO (required for LIBERO assets/tasks)

If your wrappers reference LIBERO objects (e.g. `milk`, `ketchup`, `plate`, `red_bowl`) you must install LIBERO locally so the assets and Python package are available.

#### 6.1 Clone LIBERO

From your workspace root:

```bash
cd <workspace>
git clone https://github.com/damurka5/LIBERO.git
```

#### 6.2 Install LIBERO (editable)

```bash
conda activate cdpr
pip install -e ./LIBERO
```

#### 6.3 (Recommended) Install extra common deps used by LIBERO

Some LIBERO components expect these:

```bash
pip install easydict jsonlines
pip install gym==0.26.2 gym-notices
pip install robosuite==1.4.0
```

> If you already installed `gym` / `robosuite` earlier, you can skip this.

#### 6.4 Verify LIBERO import

```bash
python - <<EOF
import libero
print("LIBERO OK")
EOF
```

---

## 7. Teleoperation dependencies

### 7.1 Install keyboard listener

```bash
pip install pynput
```

### 7.2 macOS permission (IMPORTANT)

On macOS, allow keyboard access:

* **System Settings → Privacy & Security → Accessibility**
* Enable your terminal (Terminal / iTerm / VSCode)

Restart terminal afterward.

---

## 8. Platform-specific configuration

---

### macOS (IMPORTANT)

#### 8.1 Force GLFW backend

MuJoCo EGL is Linux-only and causes crashes on macOS.

Set this **every time** or in your shell config:

```bash
export MUJOCO_GL=glfw
```

Optional (persistent):

```bash
echo 'export MUJOCO_GL=glfw' >> ~/.zshrc
```

#### 8.2 Scene switcher segfault (expected)

On macOS, `cdpr_scene_switcher` may:

* successfully write wrapper XML
* then segfault during shutdown

**This is handled by patching `build_wrapper_if_needed()`**
(see section 10).

---

### Linux

* EGL may work if GPU drivers are installed
* If EGL fails, use GLFW:

```bash
export MUJOCO_GL=glfw
```

* Headless rendering works well on Linux

---

### Windows

* Use **Anaconda Prompt**
* GLFW backend is recommended:

```bat
set MUJOCO_GL=glfw
```

* `pynput` works normally
* No EGL support

---

## 9. `.gitignore` (important)

Add to `CDPR-Dataset/.gitignore`:

```gitignore
# Generated MuJoCo wrappers (machine-specific)
cdpr_dataset/wrappers/*_wrapper.xml
cdpr_dataset/wrappers/*__localpatched.xml

# Generated datasets
cdpr_dataset/datasets/cdpr_synth/
```

If already tracked:

```bash
git rm --cached cdpr_dataset/wrappers/*.xml
```

---

## 10. Required patch for macOS stability

### Patch `build_wrapper_if_needed()`

File:

```
cdpr_dataset/generate_cdpr_dataset.py
```

**Replace**:

```python
subprocess.run(cmd, check=True)
```

**With**:

```python
print(">>", " ".join(cmd))
proc = subprocess.run(cmd)

if proc.returncode != 0:
    # macOS: scene switcher may segfault after writing wrapper
    if wrapper_out.exists() and wrapper_out.stat().st_size > 0:
        print(
            f"⚠️ cdpr_scene_switcher exited with code {proc.returncode}, "
            f"but wrapper was created. Continuing."
        )
    else:
        raise RuntimeError(
            f"cdpr_scene_switcher failed (code {proc.returncode}) and wrapper was not created."
        )

print(f"✅ Built wrapper: {wrapper_out}")
return wrapper_out
```

This is **mandatory on macOS**.

---

## 11. Synthetic dataset generation

From `CDPR-Dataset/`:

```bash
conda activate cdpr
MUJOCO_GL=glfw python -m cdpr_dataset.generate_cdpr_dataset \
  --episodes_per_scene 1 \
  --tasks put_into_bowl \
  --catalog cdpr_dataset/datasets/cdpr_scene_catalog.yaml
```

Outputs appear in:

```text
cdpr_dataset/datasets/cdpr_synth/videos/
```

---

## 12. Teleoperation (keyboard control)

### Controls

| Key     | Action        |
| ------- | ------------- |
| W / S   | +Y / −Y       |
| A / D   | −X / +X       |
| ↑ / ↓   | +Z / −Z       |
| [ / ]   | Yaw − / +     |
| X       | Open gripper  |
| C       | Close gripper |
| Q / ESC | End episode   |

### Run teleop

```bash
conda activate cdpr
MUJOCO_GL=glfw 
python -m cdpr_dataset.teleop_cdpr_collect \
  --episodes 1 \
  --catalog cdpr_dataset/datasets/cdpr_scene_catalog.yaml \
  --task_name put_into_bowl \
  --language "Put the milk into the bowl." \
  --force_rebuild_wrapper
```

Saved episodes are **identical in format** to synthetic ones and can be exported to RLDS.

---

## 13. Sanity check

```bash
python - <<EOF
import mujoco
import cdpr_mujoco
import cdpr_dataset
print("Environment OK")
EOF
```
