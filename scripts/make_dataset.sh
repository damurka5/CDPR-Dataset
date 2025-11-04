```bash
#!/usr/bin/env bash
set -euo pipefail

python -m cdpr_dataset.generate_cdpr_dataset \
  --episodes_per_scene "${EPISODES_PER_SCENE:-5}" \
  --tasks pick_and_hover push_left push_forward