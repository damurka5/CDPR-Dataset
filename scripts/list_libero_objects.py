#!/usr/bin/env python3
import os
from pathlib import Path
from cdpr_mujoco.cdpr_scene_switcher import OBJECTS_DIR

def main():
    d = Path(OBJECTS_DIR)
    if not d.exists():
        print(f"Objects dir not found: {d}")
        return
    names = []
    for p in sorted(d.iterdir()):
        if p.is_dir() and any(x.suffix == ".xml" for x in p.glob("*.xml")):
            names.append(p.name)
    print(f"Found {len(names)} LIBERO objects:")
    for n in names:
        print(n)

if __name__ == "__main__":
    main()
