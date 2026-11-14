import copy
import logging
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

import torch

import folder_paths
import comfy.utils
import comfy.sd


# =========================================================
# Layer mappings
# =========================================================
# Stage 1: legacy Anima 28L -> Anima 2.9B 40L
MAP_28_TO_40 = {
     0:  0,
     1:  1,
     2:  3,
     3:  4,
     4:  6,
     5:  7,
     6:  9,
     7: 10,
     8: 12,
     9: 13,
    10: 15,
    11: 16,
    12: 18,
    13: 19,
    14: 20,
    15: 22,
    16: 23,
    17: 25,
    18: 26,
    19: 28,
    20: 29,
    21: 31,
    22: 32,
    23: 34,
    24: 35,
    25: 37,
    26: 38,
    27: 39,
}

INSERTED_40 = [2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36]

# Stage 2: Anima 2.9B 40L -> Anima 3.8B 52L
# The 12 new 3.8B blocks occupy these final 52L positions.
INSERTED_52 = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]

# The 40 old blocks fill every non-inserted final position in order.
_ANCHOR_POSITIONS_52 = [i for i in range(52) if i not in INSERTED_52]
MAP_40_TO_52 = {old_idx: new_idx for old_idx, new_idx in enumerate(_ANCHOR_POSITIONS_52)}

# Useful direct map for reports/debugging only.
MAP_28_TO_52 = {old_idx: MAP_40_TO_52[mid_idx] for old_idx, mid_idx in MAP_28_TO_40.items()}


# =========================================================
# Key parsing helpers
# =========================================================
# Common key patterns:
#   net.blocks.12.xxx
#   diffusion_model.blocks.12.xxx
#   lora_unet_blocks_12_xxx
DOT_PATTERN = re.compile(r"(blocks\.)(\d+)(\.)")
US_PATTERN = re.compile(r"(blocks_)(\d+)(_)")

UP_PATTERNS = (".lora_up.weight", "_lora_up.weight", "lora_up.weight")


def is_llm_adapter_key(key: str) -> bool:
    return "llm_adapter" in key.lower()


def replace_block_index(key: str, new_idx) -> Tuple[str, Optional[int]]:
    if is_llm_adapter_key(key):
        return key, None

    m = DOT_PATTERN.search(key)
    if m:
        old_idx = int(m.group(2))
        return key[:m.start(2)] + str(new_idx) + key[m.end(2):], old_idx

    m = US_PATTERN.search(key)
    if m:
        old_idx = int(m.group(2))
        return key[:m.start(2)] + str(new_idx) + key[m.end(2):], old_idx

    return key, None


def get_block_index(key: str) -> Optional[int]:
    if is_llm_adapter_key(key):
        return None

    m = DOT_PATTERN.search(key)
    if m:
        return int(m.group(2))

    m = US_PATTERN.search(key)
    if m:
        return int(m.group(2))

    return None


def normalize_signature(key: str) -> str:
    out, _ = replace_block_index(key, "{BLOCK}")
    return out


def clone_tensor(x):
    if torch.is_tensor(x):
        return x.clone()
    return copy.deepcopy(x)


def blend_tensors(a, b, ratio_prev=0.5, ratio_next=0.5):
    if a is None and b is None:
        return None
    if a is None:
        return clone_tensor(b)
    if b is None:
        return clone_tensor(a)
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return clone_tensor(a)
    if a.shape != b.shape:
        return clone_tensor(a)
    return (a.float() * ratio_prev + b.float() * ratio_next).to(a.dtype)


def is_attention_related_key(key: str) -> bool:
    k = key.lower()
    return ("self_attn" in k) or ("cross_attn" in k) or ("attn" in k and "mlp" not in k)


def is_mlp_related_key(key: str) -> bool:
    return "mlp" in key.lower()


def should_include_inserted_key(key: str, inserted_scope: str) -> bool:
    scope = inserted_scope.lower()
    if scope == "all":
        return True
    if scope == "attn_only":
        return is_attention_related_key(key)
    if scope == "attn_and_mlp":
        return is_attention_related_key(key) or is_mlp_related_key(key)
    return True


def scaled_for_inserted(key: str, value, inserted_strength: float):
    if not torch.is_tensor(value):
        return value
    lk = key.lower()
    # Scaling only LoRA-up is sufficient to scale the resulting LoRA delta.
    if any(p in lk for p in UP_PATTERNS):
        return (value.float() * inserted_strength).to(value.dtype)
    return value.clone()


def select_source_tensor(
    sig: str,
    prev_map: Dict[str, Tuple[str, object]],
    next_map: Dict[str, Tuple[str, object]],
    mode: str,
):
    prev_item = prev_map.get(sig)
    next_item = next_map.get(sig)

    prev_key = prev_item[0] if prev_item else None
    prev_val = prev_item[1] if prev_item else None
    next_key = next_item[0] if next_item else None
    next_val = next_item[1] if next_item else None

    mode = mode.lower()

    if mode == "copy_prev":
        if prev_item is not None:
            return prev_key, clone_tensor(prev_val)
        if next_item is not None:
            return next_key, clone_tensor(next_val)
        return None, None

    if mode == "copy_next":
        if next_item is not None:
            return next_key, clone_tensor(next_val)
        if prev_item is not None:
            return prev_key, clone_tensor(prev_val)
        return None, None

    if mode == "blend":
        if prev_item is None and next_item is None:
            return None, None
        source_key = prev_key if prev_key is not None else next_key
        return source_key, blend_tensors(prev_val, next_val, 0.5, 0.5)

    if mode == "sharp_blend":
        if prev_item is None and next_item is None:
            return None, None
        source_key = prev_key if prev_key is not None else next_key
        return source_key, blend_tensors(prev_val, next_val, 0.7, 0.3)

    if prev_item is not None:
        return prev_key, clone_tensor(prev_val)
    if next_item is not None:
        return next_key, clone_tensor(next_val)
    return None, None


def build_inserted_neighbors(anchor_map: Dict[int, int], inserted_blocks: Iterable[int]):
    sorted_pairs = sorted(anchor_map.items(), key=lambda x: x[1])
    result = {}
    for inserted in inserted_blocks:
        prev_pair = None
        next_pair = None
        for old_idx, new_idx in sorted_pairs:
            if new_idx < inserted:
                prev_pair = (old_idx, new_idx)
            elif new_idx > inserted and next_pair is None:
                next_pair = (old_idx, new_idx)
                break
        if prev_pair is None or next_pair is None:
            raise RuntimeError(f"Could not determine neighbors for inserted block {inserted}")
        result[inserted] = (prev_pair[0], next_pair[0])
    return result


NEIGHBORS_40 = build_inserted_neighbors(MAP_28_TO_40, INSERTED_40)
NEIGHBORS_52 = build_inserted_neighbors(MAP_40_TO_52, INSERTED_52)


def remap_and_expand_stage(
    lora_sd: Dict[str, object],
    source_block_count: int,
    anchor_map: Dict[int, int],
    inserted_blocks,
    inserted_neighbors,
    inserted_mode: str = "copy_prev",
    inserted_strength: float = 0.18,
    inserted_scope: str = "attn_only",
):
    """Remap one architecture stage and optionally synthesize its inserted blocks."""
    out = {}
    collisions = []
    remapped_count = 0
    inserted_count = 0
    examples = []

    source_entries = {i: {} for i in range(source_block_count)}

    for key, value in lora_sd.items():
        old_idx = get_block_index(key)

        # Non-block keys (including llm_adapter) pass through unchanged.
        if old_idx is None:
            if key in out:
                collisions.append((key, key))
            out[key] = clone_tensor(value)
            continue

        # Unsupported/out-of-range block keys also pass through unchanged.
        if old_idx not in anchor_map:
            if key in out:
                collisions.append((key, key))
            out[key] = clone_tensor(value)
            continue

        new_idx = anchor_map[old_idx]
        new_key, _ = replace_block_index(key, new_idx)

        if new_key in out and new_key != key:
            collisions.append((key, new_key))
        else:
            out[new_key] = clone_tensor(value)
            remapped_count += 1
            if len(examples) < 14:
                examples.append(f"anchor {old_idx:02d}->{new_idx:02d}: {key} -> {new_key}")

        sig = normalize_signature(key)
        source_entries[old_idx][sig] = (key, value)

    effective_mode = "copy_prev" if inserted_mode == "disabled" else inserted_mode
    effective_strength = 0.0 if inserted_mode == "disabled" else inserted_strength

    for inserted_new in inserted_blocks:
        prev_old, next_old = inserted_neighbors[inserted_new]
        prev_map = source_entries.get(prev_old, {})
        next_map = source_entries.get(next_old, {})
        signatures = sorted(set(prev_map.keys()) | set(next_map.keys()))

        for sig in signatures:
            any_source_key = None
            if sig in prev_map:
                any_source_key = prev_map[sig][0]
            elif sig in next_map:
                any_source_key = next_map[sig][0]

            if any_source_key is None:
                continue
            if not should_include_inserted_key(any_source_key, inserted_scope):
                continue

            source_key, tensor_val = select_source_tensor(
                sig, prev_map, next_map, effective_mode
            )
            if source_key is None or tensor_val is None:
                continue

            inserted_key, _ = replace_block_index(source_key, inserted_new)
            if inserted_key in out:
                collisions.append((source_key, inserted_key))
                continue

            out[inserted_key] = scaled_for_inserted(
                inserted_key, tensor_val, effective_strength
            )
            inserted_count += 1

    if collisions:
        preview = "\n".join(f"{a} -> {b}" for a, b in collisions[:12])
        raise RuntimeError(
            "Legacy Anima LoRA remap/expansion produced collisions.\n"
            f"First collisions:\n{preview}"
        )

    return out, remapped_count, inserted_count, examples


def remap_28_to_40(
    lora_sd,
    inserted_mode="copy_prev",
    inserted_strength=0.18,
    inserted_scope="attn_only",
):
    return remap_and_expand_stage(
        lora_sd,
        source_block_count=28,
        anchor_map=MAP_28_TO_40,
        inserted_blocks=INSERTED_40,
        inserted_neighbors=NEIGHBORS_40,
        inserted_mode=inserted_mode,
        inserted_strength=inserted_strength,
        inserted_scope=inserted_scope,
    )


def remap_40_to_52(
    lora_sd,
    inserted_mode="copy_prev",
    inserted_strength=0.12,
    inserted_scope="attn_only",
):
    return remap_and_expand_stage(
        lora_sd,
        source_block_count=40,
        anchor_map=MAP_40_TO_52,
        inserted_blocks=INSERTED_52,
        inserted_neighbors=NEIGHBORS_52,
        inserted_mode=inserted_mode,
        inserted_strength=inserted_strength,
        inserted_scope=inserted_scope,
    )


def detect_legacy_28l(lora_sd) -> int:
    count = 0
    for key in lora_sd.keys():
        idx = get_block_index(key)
        if idx is not None and idx in MAP_28_TO_40:
            count += 1
    return count


# =========================================================
# v1.4 source-LoRA architecture detection
# =========================================================
ARCH_28 = "28L"
ARCH_40 = "40L"
ARCH_52 = "52L"
ARCH_AUTO = "auto"


def collect_block_indices(lora_sd):
    return sorted({
        idx for key in lora_sd.keys()
        if (idx := get_block_index(key)) is not None
    })


def read_safetensors_metadata(lora_path: str) -> Dict[str, str]:
    """Best-effort safetensors metadata read. Non-safetensors files return {}."""
    if not str(lora_path).lower().endswith(".safetensors"):
        return {}
    try:
        from safetensors import safe_open
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            return dict(f.metadata() or {})
    except Exception as e:
        logging.debug("Could not read LoRA safetensors metadata: %s", e)
        return {}


def _architecture_hint_from_text(text: str):
    """
    Return (architecture, reason) from explicit architecture text.
    Used only when block indices alone are not decisive.
    """
    t = (text or "").lower().replace("\\", "/")

    patterns_52 = [
        r"anima[\s_\-/]*3(?:[._]?8)b",
        r"\b3(?:[._]?8)b\b",
        r"\b52[\s_\-]*(?:l|layer|layers|block|blocks)\b",
        r"\b(?:l|layer|layers|block|blocks)[\s_\-]*52\b",
    ]
    patterns_40 = [
        r"anima[\s_\-/]*2(?:[._]?9)b",
        r"\b2(?:[._]?9)b\b",
        r"\b40[\s_\-]*(?:l|layer|layers|block|blocks)\b",
        r"\b(?:l|layer|layers|block|blocks)[\s_\-]*40\b",
    ]
    patterns_28 = [
        r"\b28[\s_\-]*(?:l|layer|layers|block|blocks)\b",
        r"\b(?:l|layer|layers|block|blocks)[\s_\-]*28\b",
        r"legacy[\s_\-/]*(?:anima)?[\s_\-/]*28",
    ]

    for p in patterns_52:
        if re.search(p, t):
            return ARCH_52, f"text matched /{p}/"
    for p in patterns_40:
        if re.search(p, t):
            return ARCH_40, f"text matched /{p}/"
    for p in patterns_28:
        if re.search(p, t):
            return ARCH_28, f"text matched /{p}/"
    return None, ""


def detect_lora_architecture(lora_sd, lora_path: str = "", metadata=None):
    """
    Detect whether the source LoRA was authored for 28L, 40L, or 52L Anima.

    Reliable full-LoRA rules:
      max block >= 40 -> 52L
      max block >= 28 -> 40L
      max block <= 27 -> 28L unless metadata/filename explicitly says 40L/52L

    A partial 40L/52L LoRA that touches only blocks 0..27 is mathematically
    indistinguishable from a 28L LoRA by tensor keys alone. In that case the
    node reports low confidence and supports a manual source_type override.
    """
    indices = collect_block_indices(lora_sd)
    if not indices:
        return None, "none", "No recognizable Anima main-block indices were found.", indices

    max_idx = max(indices)
    min_idx = min(indices)

    if max_idx > 51:
        return None, "none", f"Highest block index is {max_idx}; expected at most 51.", indices

    # Decisive by key topology.
    if max_idx >= 40:
        return ARCH_52, "high", f"Highest detected block index is {max_idx} (>=40).", indices
    if max_idx >= 28:
        return ARCH_40, "high", f"Highest detected block index is {max_idx} (28..39).", indices

    # Keys are all within 0..27. Check metadata and filename for explicit hints.
    metadata = metadata or {}
    if metadata:
        meta_text = " ".join(f"{k}={v}" for k, v in metadata.items())
        arch, why = _architecture_hint_from_text(meta_text)
        if arch:
            return arch, "medium", f"All tensor block indices are <=27, but safetensors metadata says {arch}: {why}", indices

    if lora_path:
        arch, why = _architecture_hint_from_text(Path(lora_path).name)
        if arch:
            return arch, "medium", f"All tensor block indices are <=27, but filename suggests {arch}: {why}", indices

        # Common Anima community shorthand.
        name = Path(lora_path).name.lower()
        if re.search(r"(?:^|[_\-.])an(?:ima)?[_\-.]*38(?:b)?(?:[_\-.]|$)", name):
            return ARCH_52, "medium", "Filename contains an Anima 3.8 / AN38-style hint.", indices
        if re.search(r"(?:^|[_\-.])an(?:ima)?[_\-.]*29(?:b)?(?:[_\-.]|$)", name):
            return ARCH_40, "medium", "Filename contains an Anima 2.9 / AN29-style hint.", indices

    # Full legacy LoRAs commonly cover most/all 0..27 blocks.
    coverage = len(set(i for i in indices if 0 <= i <= 27))
    confidence = "medium" if (max_idx >= 24 or coverage >= 20) else "low"
    reason = (
        f"All detected block indices are within 0..27 (range {min_idx}..{max_idx}, "
        f"{coverage}/28 legacy block positions present). Assuming legacy 28L."
    )
    if confidence == "low":
        reason += " This could also be a partial 40L/52L LoRA; use source_type override if needed."
    return ARCH_28, confidence, reason, indices


def validate_forced_architecture(forced_arch: str, indices):
    if not indices:
        raise RuntimeError("No recognizable Anima main-block keys were detected in this LoRA.")
    max_idx = max(indices)
    if forced_arch == ARCH_28 and max_idx > 27:
        raise RuntimeError(
            f"source_type=28L was forced, but block {max_idx} exists. "
            "A 28L LoRA must not contain main-block indices above 27."
        )
    if forced_arch == ARCH_40 and max_idx > 39:
        raise RuntimeError(
            f"source_type=40L was forced, but block {max_idx} exists. "
            "A 40L LoRA must not contain main-block indices above 39."
        )
    if forced_arch == ARCH_52 and max_idx > 51:
        raise RuntimeError(
            f"source_type=52L was forced, but block {max_idx} exists. "
            "A 52L LoRA must not contain main-block indices above 51."
        )


def block_index_summary(indices):
    if not indices:
        return "none"
    if len(indices) <= 18:
        return ", ".join(str(x) for x in indices)
    return (
        ", ".join(str(x) for x in indices[:9])
        + ", ..., "
        + ", ".join(str(x) for x in indices[-9:])
    )


# =========================================================
# Original 28L -> 40L node (kept for backward compatibility)
# =========================================================
class Anima29LegacyLoRALoaderSharp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 0.75, "min": -5.0, "max": 5.0, "step": 0.01}),
                "inserted_strength": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "inserted_mode": (["copy_prev", "sharp_blend", "blend", "copy_next", "disabled"],),
                "inserted_scope": (["attn_only", "attn_and_mlp", "all"],),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load_legacy_lora"
    CATEGORY = "Anima/Legacy LoRA"

    def load_legacy_lora(
        self,
        model,
        lora_name,
        strength_model,
        inserted_strength,
        inserted_mode,
        inserted_scope,
    ):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        original_count = len(lora_sd)

        if detect_legacy_28l(lora_sd) == 0:
            raise RuntimeError(
                "No recognizable 28-layer Anima main-block keys were detected.\n"
                "This LoRA may already be expanded or use an unsupported key format."
            )

        remapped_sd, remapped_count, inserted_count, examples = remap_28_to_40(
            lora_sd,
            inserted_mode=inserted_mode,
            inserted_strength=inserted_strength,
            inserted_scope=inserted_scope,
        )

        model_out, _ = comfy.sd.load_lora_for_models(
            model, None, remapped_sd, strength_model, 0.0
        )

        example_text = "\n".join("  " + x for x in examples[:10])
        report = (
            "Anima legacy LoRA: 28L -> 40L\n"
            f"LoRA: {lora_name}\n"
            f"LoRA strength: {strength_model}\n"
            f"40L inserted mode/strength/scope: {inserted_mode} / {inserted_strength} / {inserted_scope}\n"
            f"Original entries: {original_count}\n"
            f"Anchor-remapped entries: {remapped_count}\n"
            f"Inserted synthetic entries: {inserted_count}\n"
            "LLM adapter keys: unchanged\n"
            "Examples:\n"
            f"{example_text}"
        )
        logging.info(report)
        return (model_out, report)


# =========================================================
# New 28L -> 40L -> 52L node for Anima 3.8B
# =========================================================
class Anima38LegacyLoRALoaderSharp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 0.75, "min": -5.0, "max": 5.0, "step": 0.01}),
                "stage1_28to40_strength": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stage1_28to40_mode": (["copy_prev", "sharp_blend", "blend", "copy_next", "disabled"],),
                "stage1_28to40_scope": (["attn_only", "attn_and_mlp", "all"],),
                "stage2_40to52_strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stage2_40to52_mode": (["copy_prev", "sharp_blend", "blend", "copy_next", "disabled"],),
                "stage2_40to52_scope": (["attn_only", "attn_and_mlp", "all"],),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load_legacy_lora_52l"
    CATEGORY = "Anima 3.8B/Legacy LoRA"

    def load_legacy_lora_52l(
        self,
        model,
        lora_name,
        strength_model,
        stage1_28to40_strength,
        stage1_28to40_mode,
        stage1_28to40_scope,
        stage2_40to52_strength,
        stage2_40to52_mode,
        stage2_40to52_scope,
    ):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        original_count = len(lora_sd)

        detected_legacy = detect_legacy_28l(lora_sd)
        if detected_legacy == 0:
            raise RuntimeError(
                "No recognizable legacy 28-layer Anima main-block keys were detected.\n"
                "This node expects a 28L legacy LoRA and expands it 28L -> 40L -> 52L."
            )

        # Stage 1: 28 -> 40
        sd40, s1_anchor_count, s1_inserted_count, examples_40 = remap_28_to_40(
            lora_sd,
            inserted_mode=stage1_28to40_mode,
            inserted_strength=stage1_28to40_strength,
            inserted_scope=stage1_28to40_scope,
        )

        # Stage 2: the complete 40L result (including stage-1 synthetic blocks)
        # is then expanded into the 52L Anima 3.8B layout.
        sd52, s2_anchor_count, s2_inserted_count, examples_52 = remap_40_to_52(
            sd40,
            inserted_mode=stage2_40to52_mode,
            inserted_strength=stage2_40to52_strength,
            inserted_scope=stage2_40to52_scope,
        )

        model_out, _ = comfy.sd.load_lora_for_models(
            model, None, sd52, strength_model, 0.0
        )

        direct_examples = []
        for old_idx in range(28):
            direct_examples.append(
                f"{old_idx:02d}->{MAP_28_TO_40[old_idx]:02d}->{MAP_28_TO_52[old_idx]:02d}"
            )
        mapping_preview = ", ".join(direct_examples)

        ex40 = "\n".join("  " + x for x in examples_40[:6])
        ex52 = "\n".join("  " + x for x in examples_52[:6])

        report = (
            "Anima 3.8B legacy LoRA bridge: 28L -> 40L -> 52L\n"
            f"LoRA: {lora_name}\n"
            f"LoRA strength: {strength_model}\n"
            f"Detected legacy block entries: {detected_legacy}\n"
            "\n"
            "Stage 1 / Anima 2.9B expansion (28 -> 40)\n"
            f"  inserted blocks: {INSERTED_40}\n"
            f"  mode/strength/scope: {stage1_28to40_mode} / {stage1_28to40_strength} / {stage1_28to40_scope}\n"
            f"  anchor-remapped entries: {s1_anchor_count}\n"
            f"  synthetic entries: {s1_inserted_count}\n"
            "\n"
            "Stage 2 / Anima 3.8B expansion (40 -> 52)\n"
            f"  inserted final positions: {INSERTED_52}\n"
            f"  mode/strength/scope: {stage2_40to52_mode} / {stage2_40to52_strength} / {stage2_40to52_scope}\n"
            f"  anchor-remapped entries: {s2_anchor_count}\n"
            f"  synthetic entries: {s2_inserted_count}\n"
            "\n"
            f"Original entries: {original_count}\n"
            f"Final 52L entries: {len(sd52)}\n"
            "LLM adapter keys: unchanged\n"
            "\n"
            "Direct legacy anchor path (28 -> 40 -> 52):\n"
            f"  {mapping_preview}\n"
            "\n"
            "Stage-1 examples:\n"
            f"{ex40}\n"
            "Stage-2 examples:\n"
            f"{ex52}\n"
            "\n"
            "Anti-blur starting point:\n"
            "  strength_model=0.65~0.85\n"
            "  stage1: copy_prev, 0.12~0.22, attn_only\n"
            "  stage2: copy_prev, 0.08~0.16, attn_only\n"
        )

        logging.info(report)
        return (model_out, report)



# =========================================================
# v1.4 Universal source loader for Anima 3.8B / 52L target
# =========================================================
class Anima38AutoLoRALoader:
    """
    One-node Anima 3.8B LoRA loader.

    Source 28L -> 28->40->52
    Source 40L -> 40->52
    Source 52L -> direct load

    source_type='auto' inspects block indices first, then safetensors metadata
    and filename hints when a partial LoRA is ambiguous.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 0.75, "min": -5.0, "max": 5.0, "step": 0.01}),
                "source_type": ([ARCH_AUTO, ARCH_28, ARCH_40, ARCH_52],),
                "stage1_28to40_strength": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stage1_28to40_mode": (["copy_prev", "sharp_blend", "blend", "copy_next", "disabled"],),
                "stage1_28to40_scope": (["attn_only", "attn_and_mlp", "all"],),
                "stage2_40to52_strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stage2_40to52_mode": (["copy_prev", "sharp_blend", "blend", "copy_next", "disabled"],),
                "stage2_40to52_scope": (["attn_only", "attn_and_mlp", "all"],),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load_auto_lora"
    CATEGORY = "Anima 3.8B/Universal LoRA"

    def load_auto_lora(
        self,
        model,
        lora_name,
        strength_model,
        source_type,
        stage1_28to40_strength,
        stage1_28to40_mode,
        stage1_28to40_scope,
        stage2_40to52_strength,
        stage2_40to52_mode,
        stage2_40to52_scope,
    ):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        original_count = len(lora_sd)
        metadata = read_safetensors_metadata(lora_path)

        detected_arch, confidence, detect_reason, indices = detect_lora_architecture(
            lora_sd, lora_path=lora_path, metadata=metadata
        )

        if source_type == ARCH_AUTO:
            if detected_arch is None:
                raise RuntimeError(
                    "Could not determine the source Anima architecture automatically.\n"
                    f"{detect_reason}\n"
                    "Set source_type manually to 28L, 40L, or 52L."
                )
            source_arch = detected_arch
            source_note = f"auto -> {source_arch} ({confidence} confidence)"
        else:
            source_arch = source_type
            validate_forced_architecture(source_arch, indices)
            source_note = f"manual override -> {source_arch}"
            if detected_arch is not None and detected_arch != source_arch:
                detect_reason += (
                    f" Manual override intentionally selected {source_arch} "
                    f"instead of auto-detected {detected_arch}."
                )

        # ---------------------------------------------------------
        # Convert source LoRA to the 52L target layout.
        # ---------------------------------------------------------
        stage1_text = "not used"
        stage2_text = "not used"
        examples_text = ""

        if source_arch == ARCH_28:
            sd40, s1_anchor, s1_inserted, ex40 = remap_28_to_40(
                lora_sd,
                inserted_mode=stage1_28to40_mode,
                inserted_strength=stage1_28to40_strength,
                inserted_scope=stage1_28to40_scope,
            )
            sd52, s2_anchor, s2_inserted, ex52 = remap_40_to_52(
                sd40,
                inserted_mode=stage2_40to52_mode,
                inserted_strength=stage2_40to52_strength,
                inserted_scope=stage2_40to52_scope,
            )
            final_sd = sd52
            conversion_path = "28L -> 40L -> 52L"
            stage1_text = (
                f"28->40 anchors={s1_anchor}, synthetic={s1_inserted}, "
                f"{stage1_28to40_mode}/{stage1_28to40_strength}/{stage1_28to40_scope}"
            )
            stage2_text = (
                f"40->52 anchors={s2_anchor}, synthetic={s2_inserted}, "
                f"{stage2_40to52_mode}/{stage2_40to52_strength}/{stage2_40to52_scope}"
            )
            examples_text = (
                "28->40 examples:\n" + "\n".join("  " + x for x in ex40[:4])
                + "\n40->52 examples:\n" + "\n".join("  " + x for x in ex52[:4])
            )

        elif source_arch == ARCH_40:
            sd52, s2_anchor, s2_inserted, ex52 = remap_40_to_52(
                lora_sd,
                inserted_mode=stage2_40to52_mode,
                inserted_strength=stage2_40to52_strength,
                inserted_scope=stage2_40to52_scope,
            )
            final_sd = sd52
            conversion_path = "40L -> 52L"
            stage2_text = (
                f"40->52 anchors={s2_anchor}, synthetic={s2_inserted}, "
                f"{stage2_40to52_mode}/{stage2_40to52_strength}/{stage2_40to52_scope}"
            )
            examples_text = (
                "40->52 examples:\n" + "\n".join("  " + x for x in ex52[:8])
            )

        elif source_arch == ARCH_52:
            final_sd = lora_sd
            conversion_path = "52L -> direct"
            examples_text = "No block remapping was performed."

        else:
            raise RuntimeError(f"Unsupported source architecture: {source_arch}")

        model_out, _ = comfy.sd.load_lora_for_models(
            model, None, final_sd, strength_model, 0.0
        )

        metadata_keys = ", ".join(sorted(metadata.keys())[:12]) if metadata else "none"
        report = (
            "Anima 3.8B Universal LoRA Loader v1.4\n"
            f"LoRA: {lora_name}\n"
            f"Strength: {strength_model}\n"
            f"Source selection: {source_note}\n"
            f"Auto-detected architecture: {detected_arch or 'unknown'}\n"
            f"Detection confidence: {confidence}\n"
            f"Detection reason: {detect_reason}\n"
            f"Detected block indices: {block_index_summary(indices)}\n"
            f"Conversion path: {conversion_path}\n"
            f"Original entries: {original_count}\n"
            f"Final entries loaded: {len(final_sd)}\n"
            f"Stage 1: {stage1_text}\n"
            f"Stage 2: {stage2_text}\n"
            f"Safetensors metadata keys: {metadata_keys}\n"
            "LLM adapter / non-block keys: passed through unchanged\n"
            "\n"
            f"{examples_text}\n"
        )

        if source_type == ARCH_AUTO and confidence == "low":
            report += (
                "\nWARNING: Low-confidence auto detection. This LoRA only exposes "
                "block indices <=27 and has no decisive metadata/filename hint. "
                "If it is a partial 40L or 52L LoRA, set source_type manually.\n"
            )

        logging.info(report)
        return (model_out, report)



# =========================================================
# ComfyUI registration / compatibility aliases
# =========================================================
NODE_CLASS_MAPPINGS = {
    # Existing 2.9B IDs remain unchanged for old workflows.
    "Anima29LegacyLoRALoader": Anima29LegacyLoRALoaderSharp,
    "Anima29LegacyLoRALoaderSharp": Anima29LegacyLoRALoaderSharp,

    # v1.3 dedicated legacy 28L -> 40L -> 52L bridge.
    "Anima38LegacyLoRALoaderSharp": Anima38LegacyLoRALoaderSharp,
    "Anima38LegacyLoRALoader": Anima38LegacyLoRALoaderSharp,

    # v1.4 universal source loader for Anima 3.8B / 52L targets.
    "Anima38AutoLoRALoader": Anima38AutoLoRALoader,
    "Anima38UniversalLoRALoader": Anima38AutoLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anima29LegacyLoRALoader": "Anima 2.9B Legacy 28L -> 40L LoRA Loader (Sharp)",
    "Anima29LegacyLoRALoaderSharp": "Anima 2.9B Legacy 28L -> 40L LoRA Loader (Sharp)",
    "Anima38LegacyLoRALoaderSharp": "Anima 3.8B Legacy LoRA 28L -> 40L -> 52L (Sharp)",
    "Anima38LegacyLoRALoader": "Anima 3.8B Legacy LoRA 28L -> 40L -> 52L (Sharp)",
    "Anima38AutoLoRALoader": "Anima 3.8B Universal LoRA Loader (Auto 28L / 40L / 52L)",
    "Anima38UniversalLoRALoader": "Anima 3.8B Universal LoRA Loader (Auto 28L / 40L / 52L)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
