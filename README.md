# LightControl

Utilities for generating lighting-controlled training data with Blender.

## Blender Dataset Generator

`blendify/dataset_generator.py` automates rendering RGB, object mask, and simple
shadow passes for every model in a directory. Each output variation applies
distinct, well-spaced lighting setups and ensures the object rests flush on the
ground plane via collision-aware placement.

### Requirements

- Blender 3.x with the Cycles renderer
- Python packages shipped with Blender (no external `pip` modules required)

### Usage

```bash
blender --background --python blendify/dataset_generator.py -- \
		--models /path/to/models \
		--output /path/to/output \
		--samples 128
```

Arguments:

- `--models`: Directory containing `.obj`, `.fbx`, or `.blend` assets.
- `--output`: Destination folder for generated `rgb_*.png`, `mask_*.png`, and
	`shadow_*.png` files.
- `--samples`: Cycles sample count per render (default `128`).

### Output Layout

For each asset the script produces `VARIATIONS_PER_MODEL` (default `8`) sets of
images:

- `rgb_XXXXX.png`: Render with randomized, normalized lighting.
- `mask_XXXXX.png`: Binary foreground mask.
- `shadow_XXXXX.png`: Direct shadow catch pass.

Metadata such as transform reset, helper cleanup, physics settle, and lighting
parameters are handled internally—no manual scene preparation needed.