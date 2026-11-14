# ComfyUI-Anima29-LegacyLoRA v1.4.1

Universal Anima LoRA compatibility loader for an **Anima 3.8B / 52-block target**.


## v1.4.1 Fix

v1.4.1 fixes a crash in `Anima38AutoLoRALoader` automatic architecture detection:

`NameError: name 'Path' is not defined`

This update only adds the missing Python import required by filename-based detection. **The 28L / 40L / 52L remapping behavior and LoRA weight handling are unchanged.**

**If you installed v1.4, please replace it with v1.4.1.**

## Universal Auto Loader

The new node automatically recognizes the source LoRA generation:

- legacy **28L** -> `28 -> 40 -> 52`
- Anima 2.9B **40L** -> `40 -> 52`
- Anima 3.8B **52L** -> direct load

Node:

`Anima38AutoLoRALoader`

Display name:

**Anima 3.8B Universal LoRA Loader (Auto 28L / 40L / 52L)**

The detector checks block indices first, then safetensors metadata and filename hints if needed.

A manual `source_type` override (`auto / 28L / 40L / 52L`) is included because a partial 40L or 52L LoRA containing only early blocks can be structurally ambiguous.

Recommended starting values:

- 28->40 inserted: `copy_prev`, `0.18`, `attn_only`
- 40->52 inserted: `copy_prev`, `0.12`, `attn_only`

Existing 2.9B and 3.8B legacy loader nodes remain available for backward compatibility.

No model or LoRA weights are included.
