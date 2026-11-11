"""Entry-state interventions on unmodified Transformers Qwen2 decoder blocks.

No agent runtime, learned correction, compression, or network transport. E is
the residual stream immediately before a decoder block's input RMSNorm; K is
post-RoPE. CPU snapshots are immutable inputs; every continuation gets a fresh
DynamicCache. The caller must provide identical X tokens and position IDs.
"""
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class Snapshot:
    entries: dict
    kv: tuple
    prefix_length: int
    shared_length: int


def cache_from_tensors(kv, device):
    return DynamicCache.from_legacy_cache(tuple(
        (k.to(device).clone(), v.to(device).clone()) for k, v in kv))


def relative_error(actual, expected):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def additive_entry(source, source_target_context, target_source_context):
    """Apply a context delta in FP32; BF16 addition/subtraction can erase E_Bs.

    Group the identical-context subtraction first so P0 reduces EXACTLY to
    E_Bs. Otherwise rounding noise can masquerade as cross-factor interaction.
    """
    if source.shape != source_target_context.shape or source.shape != target_source_context.shape:
        raise ValueError("All oracle entries must have the same shape")
    delta=source_target_context.float()-source.float()
    return (target_source_context.float()+delta).to(target_source_context.dtype)


def require_compatible(model_a, model_b):
    fields = ("model_type", "num_hidden_layers", "hidden_size", "num_attention_heads",
              "num_key_value_heads", "rope_theta", "vocab_size")
    for name in fields:
        if getattr(model_a.config, name) != getattr(model_b.config, name):
            raise ValueError(f"This direct-entry experiment requires equal {name}")
    if model_a.config.model_type != "qwen2" or model_a.config.use_sliding_window:
        raise ValueError("Only full-attention Qwen2 is validated by this harness")


@torch.inference_mode()
def capture(model, prefix_ids, shared_ids, entry_layers):
    if not prefix_ids or not shared_ids:
        raise ValueError("Both prefix and shared segment must be nonempty")
    p, n = len(prefix_ids), len(shared_ids)
    entries, hooks = {}, []
    layers = model.model.layers
    if any(i < 0 or i >= len(layers) for i in entry_layers):
        raise ValueError("Invalid entry layer")
    for i in entry_layers:
        def hook(_module, args, index=i):
            entries[index] = args[0][:, p:p+n].detach().cpu().clone()
        hooks.append(layers[i].register_forward_pre_hook(hook))
    try:
        ids = torch.tensor([prefix_ids + shared_ids], device=model.device)
        out = model.model(input_ids=ids, use_cache=True)
        kv = tuple((k.detach().cpu().clone(), v.detach().cpu().clone())
                   for k, v in out.past_key_values.to_legacy_cache())
    finally:
        for hook in hooks:
            hook.remove()
    return Snapshot(entries, kv, p, n)


@torch.inference_mode()
def rebuild_interval(model, target, entry, start, width):
    """Execute only [start,start+width), with the target's exact prefix KV."""
    p, n = target.prefix_length, target.shared_length
    if not (0 <= start < start + width <= len(target.kv)):
        raise ValueError("Invalid recomputation interval")
    if tuple(entry.shape) != (1, n, model.config.hidden_size):
        raise ValueError("Entry shape does not match shared tokens / hidden width")
    prefix = tuple((k[:, :, :p], v[:, :, :p]) for k, v in target.kv)
    cache = cache_from_tensors(prefix, model.device)
    h = entry.to(device=model.device, dtype=model.dtype)
    positions = torch.arange(p, p+n, device=model.device)
    # Explicit offset causal mask: every query sees H_t and only preceding X.
    allowed = torch.arange(p+n, device=model.device)[None, :] <= positions[:, None]
    mask = torch.zeros((n, p+n), dtype=model.dtype, device=model.device)
    mask.masked_fill_(~allowed, torch.finfo(model.dtype).min)
    mask = mask[None, None]
    rope = model.model.rotary_emb(h, positions[None])
    for i in range(start, start+width):
        h = model.model.layers[i](
            h, attention_mask=mask, position_ids=positions[None],
            past_key_values=cache, use_cache=True, cache_position=positions,
            position_embeddings=rope)
    result = {}
    all_kv = cache.to_legacy_cache()
    for i in range(start, start+width):
        k, v = all_kv[i]
        if k.shape[-2] != p+n:
            raise AssertionError("Interval recomputation produced a wrong cache length")
        result[i] = (k[:, :, p:].detach().cpu().clone(),
                     v[:, :, p:].detach().cpu().clone())
    return result


def assemble(source, target, replacements=None, native_outside=False):
    if source.shared_length != target.shared_length:
        raise ValueError("Shared token length differs")
    if source.prefix_length != target.prefix_length:
        raise ValueError("Position changes are deliberately excluded from stage A")
    if len(source.kv) != len(target.kv):
        raise ValueError("Layer count differs")
    replacements = replacements or {}
    p = target.prefix_length
    result = []
    for i, ((sk, sv), (tk, tv)) in enumerate(zip(source.kv, target.kv)):
        xk, xv = (tk[:, :, p:], tv[:, :, p:]) if native_outside else (sk[:, :, p:], sv[:, :, p:])
        xk, xv = replacements.get(i, (xk, xv))
        if xk.shape != tk[:, :, p:].shape or xv.shape != tv[:, :, p:].shape:
            raise ValueError("Replacement KV shape differs")
        result.append((torch.cat((tk[:, :, :p], xk), dim=-2),
                       torch.cat((tv[:, :, :p], xv), dim=-2)))
    return tuple(result)


@torch.inference_mode()
def greedy(model, kv, suffix_ids, max_new_tokens=32, stop_ids=()):
    if not suffix_ids or max_new_tokens < 1:
        raise ValueError("Need suffix and a positive decode budget")
    cache = cache_from_tensors(kv, model.device)
    current = torch.tensor([suffix_ids], device=model.device)
    tokens = []
    for _ in range(max_new_tokens):
        out = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
        token = int(out.logits[0, -1].argmax().item())
        tokens.append(token)
        if token in stop_ids:
            break
        current = torch.tensor([[token]], device=model.device)
        cache = out.past_key_values
    return tokens


@torch.inference_mode()
def continuation_logits(model, kv, suffix_ids, answer_ids):
    if not suffix_ids or not answer_ids:
        raise ValueError("Need suffix and a fixed answer sequence")
    cache = cache_from_tensors(kv, model.device)
    inputs = torch.tensor([suffix_ids + answer_ids[:-1]], device=model.device)
    out = model(input_ids=inputs, past_key_values=cache, use_cache=True,
                logits_to_keep=len(answer_ids))
    return out.logits[0].detach().float().cpu()


def distribution_metrics(logits, reference_logits, answer_ids):
    if logits.shape != reference_logits.shape or logits.shape[0] != len(answer_ids):
        raise ValueError("Teacher-forced trajectories must be aligned")
    lp = reference_logits.log_softmax(-1)
    lq = logits.log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    idx = torch.tensor(answer_ids)[:, None]
    return {"kl_reference_to_variant": kl.mean().item(),
            "nll_on_reference_continuation": -lq.gather(-1, idx).mean().item(),
            "argmax_agreement": (logits.argmax(-1) == reference_logits.argmax(-1)).float().mean().item()}
