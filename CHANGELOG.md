# Changelog

## v1.4.1
- Fixed `Anima38AutoLoRALoader` auto-detection crash: `NameError: name 'Path' is not defined`.
- Added the missing `from pathlib import Path` import used by filename-based architecture detection.
- No 28L / 40L / 52L remapping logic, propagation behavior, or LoRA weight handling was changed.
- Recommended update for all v1.4 users.


## v1.4
- Added `Anima38AutoLoRALoader` / `Anima38UniversalLoRALoader`.
- Added automatic source LoRA detection for 28L, 40L, and 52L Anima LoRAs.
- 28L input: sequential `28L -> 40L -> 52L`.
- 40L input: `40L -> 52L`.
- 52L input: direct load with no block remapping.
- Added safetensors metadata and filename hints for ambiguous partial LoRAs.
- Added `source_type` manual override: `auto`, `28L`, `40L`, `52L`.
- Added detection confidence and conversion details to the report output.
- Retained all v1.3 / v1.2 node IDs for backward compatibility.

## v1.3
- Added `Anima38LegacyLoRALoaderSharp` / `Anima38LegacyLoRALoader`.
- Added sequential legacy LoRA expansion: **28L -> 40L -> 52L**.
- Added Anima 3.8B 40L -> 52L insertion positions: `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47`.
- Added independent stage-1 and stage-2 mode / strength / scope controls.
- Kept original Anima 2.9B 28L -> 40L node IDs and behavior for backward compatibility.
- Added anti-blur stage-2 default of `0.12`, `copy_prev`, `attn_only`.

## v1.2
- Added Sharp / anti-blur compatibility behavior.
- Added weak LoRA propagation to inserted 2.9B blocks.
- Added `copy_prev`, `sharp_blend`, `blend`, `copy_next`, and `disabled`.
- Added `attn_only`, `attn_and_mlp`, and `all` scopes.
- Added compatibility aliases for both old and new 2.9B node IDs.

## v1.1
- Fixed ComfyUI custom-node folder packaging.

## v1.0
- Initial 28-layer -> 40-layer legacy LoRA remapping loader.
