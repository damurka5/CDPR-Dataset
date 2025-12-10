#!/usr/bin/env python3
import os
from pathlib import Path
from cdpr_mujoco.cdpr_scene_switcher import OBJECTS_DIRS

def main():
    names = set()
    for d in OBJECTS_DIRS:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_dir() and any(x.suffix == ".xml" for x in p.glob("*.xml")):
                names.add(p.name)

    names = sorted(names)
    print(f"Found {len(names)} LIBERO objects:")
    for n in names:
        print(n)

if __name__ == "__main__":
    main()
