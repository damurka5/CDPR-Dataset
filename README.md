# CDPR Dataset Generator

This repository provides a **synthetic dataset generation pipeline** for the Cable-Driven Parallel Robot (CDPR) simulation in MuJoCo.  
It builds automatically annotated episodes compatible with **RLDS** / **Open-X Embodiment** style datasets, ready for use with **OpenVLA-OFT** or other visuomotor transformer models.

---

## 🧩 Repository Overview

```

cdpr_dataset/
├── pyproject.toml
├── requirements.txt
├── README.md
├── cdpr_dataset/
│   ├── generate_cdpr_dataset.py      # main dataset builder
│   ├── synthetic_tasks.py            # scripted trajectories (pick, push, etc.)
│   ├── datasets/
│   │   ├── cdpr_scene_catalog.yaml   # scene/object definitions
│   │   └── cdpr_synth/               # generated outputs
│   └── wrappers/                     # auto-generated scene wrappers
└── scripts/
├── make_dataset.sh               # CLI helper
├── list_libero_objects.py        # lists available objects
└── list_libero_scenes.py         # lists available scenes
```

---

## ⚙️ Installation

1. **Clone this repo** and ensure your main robot package is installed:
   ```bash
   git clone https://github.com/your-org/CDPR_Dataset.git
   cd CDPR_Dataset
   pip install -e .

2. Ensure your environment includes:

   * `openvla-oft` (for consistency with training)
   * your **installed CDPR simulator**:

     ```bash
     pip install -e /root/repo/VLA_CDPR/cdpr_mujoco
     ```

3. (Optional) If you want TFRecord export:

   ```bash
   pip install tensorflow rlds Pillow
   ```

---

## 🧠 Dataset Generation

Run the main generator:

```bash
python -m cdpr_dataset.generate_cdpr_dataset \
  --episodes_per_scene 2 \
  --tasks pick_and_hover push_left
```

Outputs are stored under:

```
cdpr_dataset/datasets/cdpr_synth/
├── npz/                 # one .npz per episode (RGB frames + actions)
├── videos/              # per-episode overview & end-effector videos
│   ├── desk_pick_and_hover_00000/
│   │   ├── overview_video.mp4
│   │   ├── ee_camera_video.mp4
│   │   └── trajectory_data.npz
│   └── desk_push_left_00001/ ...
├── meta_dataset.json    # dataset metadata
└── tfrecords/           # (optional) RLDS TFRecord shard
```

---

## 🎥 Saving Per-Episode Videos

By default, `generate_cdpr_dataset.py` saves a video for each trajectory:

* `overview_video.mp4` — fixed external camera
* `ee_camera_video.mp4` — wrist-mounted camera
* `trajectory_data.npz` — saved end-effector poses, timestamps, and internal logs

If you experience frame contamination across episodes (rare), run with:

```bash
--reinit_each_episode
```

This reinitializes the MuJoCo simulator per episode to ensure clean recordings.

---

## 🧾 Scene and Object Catalogs

Scene/object definitions live in:

```
cdpr_dataset/datasets/cdpr_scene_catalog.yaml
```

Example:

```yaml
defaults:
  scene_z: -0.85
  ee_start: [0.0, 0.0, 0.25]
  table_z: 0.15
  object_dynamic: true
  settle_time: 1.0

scenes:
  - name: desk
    objects: [ketchup, orange_juice, milk]
```

Use the helper scripts to explore your available LIBERO assets:

```bash
python scripts/list_libero_objects.py
python scripts/list_libero_scenes.py
```

---

## 📦 Dataset Structure

Each `.npz` episode contains a list of steps with keys:

| Key                                  | Description                               |
| ------------------------------------ | ----------------------------------------- |
| `observations/full_image`            | Overview camera RGB (HWC, uint8)          |
| `observations/wrist_image`           | Wrist-mounted camera RGB (HWC, uint8)     |
| `observations/state`                 | 8D normalized proprioceptive vector       |
| `language_instruction`               | task description (string)                 |
| `action/abs_7`                       | absolute Cartesian + yaw + gripper vector |
| `action/delta_7`                     | delta from previous step                  |
| `is_first`, `is_last`, `is_terminal` | step flags                                |
| `discount`, `reward`                 | scalar floats                             |

---

## 🧮 Tasks Implemented

Scripted trajectories in `synthetic_tasks.py`:

| Task             | Description                                    |
| ---------------- | ---------------------------------------------- |
| `pick_and_hover` | Move to object, close gripper, lift, and hover |
| `push_left`      | Push object along -X direction                 |
| `push_forward`   | Push object along +Y direction                 |

You can extend this by defining new motion scripts in `synthetic_tasks.py` and referencing them in `--tasks`.

---

## 🚀 Extending the Dataset

* Add new scenes or objects in `cdpr_scene_catalog.yaml`.
* Drop additional object folders under LIBERO assets or add external OBJ meshes via a converter (YCB, ShapeNet, etc.).
* Each added object is prefixed automatically in the wrapper XML to avoid asset-name collisions (`textured_vis`, etc.).

---

## 💡 Tips

* Use `--strict_objects` to enforce that all listed objects exist (fail fast).
* Use `--reinit_each_episode` to guarantee isolated video clips.
* Videos are rendered through EGL; ensure your container or machine supports GPU EGL context.
* Generated wrappers are cached in `cdpr_dataset/wrappers/`.

---

## 🧱 Example End-to-End Command

```bash
python -m cdpr_dataset.generate_cdpr_dataset \
  --episodes_per_scene 3 \
  --tasks pick_and_hover push_left push_forward \
  --reinit_each_episode
```