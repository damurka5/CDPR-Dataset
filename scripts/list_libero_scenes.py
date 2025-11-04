#!/usr/bin/env python3
from pathlib import Path
from cdpr_mujoco.cdpr_scene_switcher import SCENES_DIR

def main():
    d = Path(SCENES_DIR)
    print(f"Scenes dir: {d}")
    if not d.exists():
        print("Not found.")
        return
    names = []
    for p in sorted(d.iterdir()):
        if p.is_dir():
            xml = p / f"{p.name}.xml"
            if xml.exists():
                names.append(p.name)
    print(f"Found {len(names)} scenes:")
    for n in names:
        print(n)

if __name__ == "__main__":
    main()