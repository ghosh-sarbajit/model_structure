"""Generate a readable LaTeX architecture report for each model.

Every model is instantiated on the meta device from its config.json (no weights are read,
no GPU needed). Repeated layers are compressed into loops; the full per-tensor table is kept
in the appendix.
"""
import glob, json, os, re, struct, sys
from collections import OrderedDict

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

OUT_DIR = "/workspace/tex_output"
# /models is the host directory behind the `model` alias in ~/.bashrc
# (/opt/dlami/nvme/sarbajit/local_models), mounted into the container.
models_info = [
    ("Llama-3.2-1B", "/models/memorization/Llama-3.2-1B"),
    ("Qwen3-0.6B", "/models/Qwen3-0.6B"),
    ("Qwen3.5-0.8B", "/models/memorization/Qwen3.5-0.8B"),
    ("gemma-4-E2B", "/models/memorization/gemma-4-E2B"),
    ("Llama-3.1-8B-Instruct", "/models/Llama-3.1-8B-Instruct"),
    ("Qwen2.5-0.5B-Instruct", "/models/sayak_da_work/Qwen2.5-0.5B-Instruct"),
    ("Qwen3.5-9B", "/models/sayak_da_work/Qwen3.5-9B"),
    ("gemma-3-12b-it", "/models/gemma-3-12b-it"),
    ("gpt-oss-20b", "/models/gpt-oss-20b"),
]
# Fine-tuned variants whose architecture is identical to a model above. They get no PDF of their own;
# the main model's PDF names them (after checking class, tensor names and shapes really match).
same_arch = {
    "Llama-3.2-1B": [("Llama-3.2-1B-Instruct", "/models/sayak_da_work/Llama-3.2-1B-Instruct")],
}


# ---------------------------------------------------------------------------
# Loading and raw collection
# ---------------------------------------------------------------------------

def build_model(path):
    """Instantiate the full architecture (incl. vision/audio towers) on the meta device."""
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    cls = None
    for arch in cget(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            break
    dtype = cget(config, "dtype", None) or cget(config, "torch_dtype", None) or torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    with torch.device("meta"):
        if cls is not None:
            model = cls._from_config(config, dtype=dtype)
        else:
            model = AutoModelForCausalLM.from_config(config, dtype=dtype, trust_remote_code=True)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model, config, type(model).__name__


def collect_records(model):
    """One record per (module, parameter); tied parameters point at their first owner."""
    recs, seen, types = [], {}, {}
    for mname, mod in model.named_modules(remove_duplicate=False):
        types[mname] = type(mod).__name__
        for pname, p in mod.named_parameters(recurse=False):
            path = f"{mname}.{pname}" if mname else pname
            recs.append({
                "path": path, "module": mname, "pname": pname,
                "type": type(mod).__name__,
                "shape": list(p.shape),
                "dtype": str(p.dtype).replace("torch.", ""),
                "numel": p.numel(),
                "tied_to": seen.get(id(p)),
            })
            seen.setdefault(id(p), path)
    return recs, types


def checkpoint_only_tensors(path, types):
    """Tensors stored in the checkpoint under a top-level name the transformers class never builds
    (e.g. Qwen3.5's multi-token-prediction head `mtp.*`). Shapes come from safetensors headers."""
    known = set()
    for m in types:
        known.update(m.split("."))
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for k, v in header.items():
            if k == "__metadata__" or k.split(".")[0] in known:
                continue
            numel = 1
            for d in v["shape"]:
                numel *= d
            out.append({"path": k, "shape": v["shape"], "dtype": v["dtype"].lower(), "numel": numel})
    return sorted(out, key=lambda r: natural_key(r["path"]))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def cget(cfg, key, default=None):
    """getattr for configs; heterogeneous configs (Gemma-4) raise on per-layer attributes."""
    try:
        return getattr(cfg, key, default)
    except Exception:
        return cfg.to_dict().get(key, default)


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def esc(s):
    s = str(s)
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#",
           "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in s)


def tt(s):
    """Monospace path that may break at dots."""
    return r"\texttt{" + esc(s).replace(".", r".\allowbreak{}") + "}"


def ttype(t):
    """Class name that may break between CamelCase words."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", r"\\allowbreak{}", esc(t))


def shape_tex(shape):
    return "$" + r" \times ".join(str(d) for d in shape) + "$" if shape else "scalar"


def fmt(n):
    return f"{n:,}"


def human(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def norm_path(p):
    return re.sub(r"\.\d+(?=\.|$)", ".N", p)


def ranges_text(idx):
    """[0,1,2,4,8,12] -> '0--2, 4, 8, 12'; arithmetic progressions -> 'a, b, ..., z (every k-th)'."""
    if len(idx) >= 3:
        step = idx[1] - idx[0]
        if step > 1 and all(b - a == step for a, b in zip(idx, idx[1:])):
            return f"{idx[0]}, {idx[1]}, \\ldots, {idx[-1]} (every {step}th)"
    out, start = [], idx[0]
    for a, b in zip(idx, idx[1:] + [None]):
        if b != a + 1:
            out.append(str(start) if start == a else f"{start}--{a}")
            start = b
    return ", ".join(out)


ACT = {"silu": r"\op{SiLU}", "swish": r"\op{SiLU}", "gelu": r"\op{GELU}",
       "gelu_pytorch_tanh": r"\op{GELU}_{\tanh}", "gelu_new": r"\op{GELU}_{\tanh}", "relu": r"\op{ReLU}"}


def act_of(cfg, default="silu"):
    a = cget(cfg, "hidden_act", None) or cget(cfg, "hidden_activation", None) or default
    return ACT.get(a, r"\op{" + esc(a) + "}")


# ---------------------------------------------------------------------------
# Structure analysis: find repeated stacks and compress them into loops
# ---------------------------------------------------------------------------

def find_stacks(model, types):
    """Outermost ModuleLists whose children are numbered layers."""
    stacks = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) < 2:
            continue
        if not all(k.isdigit() for k, _ in mod.named_children()):
            continue
        if any(name.startswith(s + ".") for s in stacks):
            continue
        stacks.append(name)
    return stacks


def stack_role(stack, types):
    parent = stack.rsplit(".", 1)[0] if "." in stack else ""
    if "audio" in stack:
        return "audio"
    if "vision" in stack or "visual" in stack:
        return "vision"
    if f"{parent}.embed_tokens" in types or "embed_tokens" in types:
        return "text"
    return "other"


def role_config(config, role):
    if role == "text":
        return config.get_text_config()
    sub = cget(config, f"{role}_config", None)
    return sub if sub is not None else config


ROLE_TITLE = {"text": "Language-model decoder", "vision": "Vision encoder",
              "audio": "Audio encoder", "other": "Repeated stack"}
LT_CODE = {"sliding_attention": ("S", "Sliding-window attention block"),
           "full_attention": ("F", "Full (global) attention block"),
           "linear_attention": ("L", "Linear-attention (Gated DeltaNet) block")}


def analyse_stack(model, config, stack, recs, types):
    role = stack_role(stack, types)
    cfg = role_config(config, role)
    n = len(model.get_submodule(stack))
    layer_types = cget(cfg, "layer_types", None) if role == "text" else None
    if layer_types is not None and len(layer_types) != n:
        layer_types = None
    per_layer = [[] for _ in range(n)]
    for r in recs:
        m = re.match(re.escape(stack) + r"\.(\d+)\.(.+)$", r["path"])
        if m:
            per_layer[int(m.group(1))].append(dict(r, rel=m.group(2)))
    sigs = []
    for i in range(n):
        lt = layer_types[i] if layer_types else None
        sigs.append((lt, tuple((r["rel"], tuple(r["shape"])) for r in per_layer[i])))

    blocks = OrderedDict()   # signature -> block info
    for i, s in enumerate(sigs):
        if s not in blocks:
            blocks[s] = {"layers": [], "lt": s[0], "recs": per_layer[i], "first": i}
        blocks[s]["layers"].append(i)

    # Short code + descriptive name for each distinct block type.
    base_default = {"text": ("D", "Decoder block"), "vision": ("V", "Vision encoder block"),
                    "audio": ("A", "Conformer audio block"), "other": ("X", "Block")}
    by_code = {}
    for b in blocks.values():
        code, name = LT_CODE.get(b["lt"], base_default[role])
        b["code"], b["name"] = code, name
        by_code.setdefault(code, []).append(b)
    all_rel = [set(r["rel"] for r in b["recs"]) for b in blocks.values()]
    for code, bs in by_code.items():
        if len(bs) == 1:
            continue
        for k, b in enumerate(bs, 1):
            b["code"] = f"{code}{k}"
            rel = set(r["rel"] for r in b["recs"])
            extras = []
            if not any("k_proj" in x for x in rel) and any(any("k_proj" in x for x in o) for o in all_rel):
                extras.append("KV-shared")
            up = [r["shape"][0] for r in b["recs"] if r["rel"].endswith("up_proj.weight")]
            ups = {r["shape"][0] for bb in bs for r in bb["recs"] if r["rel"].endswith("up_proj.weight")}
            if up and len(ups) > 1:
                extras.append(f"MLP width {up[0]}")
            b["name"] += " (" + ", ".join(extras) + ")" if extras else f" (variant {k})"
    codes = [None] * n
    for b in blocks.values():
        for i in b["layers"]:
            codes[i] = b["code"]
    return {"path": stack, "role": role, "cfg": cfg, "n": n, "blocks": list(blocks.values()),
            "codes": codes, "groups": compress(codes)}


def compress(seq):
    """Greedy loop detection: returns [(unit, repeat, start_index)] covering seq."""
    out, i, n = [], 0, len(seq)
    while i < n:
        best = (1, 1)
        for p in range(1, (n - i) // 2 + 1):
            k = 1
            while i + (k + 1) * p <= n and seq[i + k * p:i + (k + 1) * p] == seq[i:i + p]:
                k += 1
            if k >= 2 and p * k > best[0] * best[1]:
                best = (p, k)
        p, k = best
        if k == 1 and out and out[-1][1] == 1:
            out[-1] = (out[-1][0] + [seq[i]], 1, out[-1][2])
        else:
            out.append((list(seq[i:i + p]), k, i))
        i += p * k
    return out


def runs(unit):
    out = []
    for c in unit:
        if out and out[-1][0] == c:
            out[-1][1] += 1
        else:
            out.append([c, 1])
    return out


def B(code):
    return r"\mathcal{B}_{\mathrm{" + code + "}}"


def stack_expr(groups):
    """Composition (right-most applied first): (B_F o B_S^{o4})^{o3} o ..."""
    parts = []
    for unit, k, _ in groups:
        inner = []
        for c, r in runs(unit):
            inner.append(B(c) + (r"^{\circ " + str(r) + "}" if r > 1 else ""))
        inner = r" \circ ".join(reversed(inner))
        if k > 1:
            inner = (r"\big(" + inner + r"\big)" if len(runs(unit)) > 1 else inner) + r"^{\circ " + str(k) + "}"
            if len(runs(unit)) == 1 and runs(unit)[0][1] > 1:
                c, r = runs(unit)[0]
                inner = B(c) + r"^{\circ " + str(r * k) + "}"
        parts.append(inner)
    return r" \circ ".join(reversed(parts))


# ---------------------------------------------------------------------------
# Block semantics: features, equations and pseudocode
# ---------------------------------------------------------------------------

def block_features(model, stack, block, role, cfg, types):
    rel = [r["rel"] for r in block["recs"]]
    shp = {r["rel"]: r["shape"] for r in block["recs"]}
    has = lambda s: any(s in x for x in rel)
    get = lambda *ks: next((shp[x] for k in ks for x in rel if x.endswith(k)), None)
    mod = model.get_submodule(f"{stack}.{block['first']}")
    attn = getattr(mod, "self_attn", None) or getattr(mod, "attn", None)
    prefix = stack.rsplit(".layers", 1)[0].rsplit(".blocks", 1)[0]
    f = {"role": role, "lt": block["lt"]}
    f["conformer"] = has("feed_forward1.")
    f["mixer"] = "gdn" if has("linear_attn.") else ("attn" if (has("self_attn.") or has("attn.")) else None)
    f["sandwich"] = has("pre_feedforward_layernorm")
    f["ln"] = has("layer_norm1") or (has("norm1.") and has("norm1.bias"))
    f["ffn"] = ("moe" if has("mlp.router") else "gated" if has("mlp.gate_proj")
                else "plain" if (has("mlp.fc1") or has("mlp.linear_fc1")) else None)
    f["ple"] = has("per_layer_input_gate")
    f["layer_scalar"] = getattr(mod, "layer_scalar", None) is not None
    f["bias"] = has("q_proj.bias") or has("qkv.bias")
    f["o_bias"] = has("o_proj.bias") or has("out_proj.bias") or has("attn.proj.bias")
    f["qk_norm"] = has("q_norm")
    f["v_norm"] = attn is not None and getattr(attn, "v_norm", None) is not None
    f["sinks"] = has("sinks")
    f["kv_shared"] = f["mixer"] == "attn" and role == "text" and not has("k_proj")
    f["fused_qkv"] = has("attn.qkv")
    f["causal"] = role == "text"
    f["rope"] = any("Rotary" in t for m, t in types.items() if m.startswith(prefix)) or role == "text"
    f["sliding"] = block["lt"] == "sliding_attention"
    f["window"] = cget(cfg, "sliding_window", None)
    f["act"] = act_of(cfg)
    f["clipped"] = any(r.endswith(".linear.weight") for r in rel)

    norm_w = get("input_layernorm.weight", "layer_norm1.weight", "norm1.weight", "norm_pre_attn.weight")
    f["d"] = norm_w[0] if norm_w else cget(cfg, "hidden_size", None)
    H = cget(cfg, "num_attention_heads", None) or cget(cfg, "num_heads", None)
    f["H"] = H
    if f["mixer"] == "attn":
        q = get("q_proj.weight", "q_proj.linear.weight")
        qkv = get("attn.qkv.weight")
        q_out = q[0] if q else (qkv[0] // 3 if qkv else None)
        qn = get("q_norm.weight")
        hd = qn[0] if qn else (cget(cfg, "head_dim", None) or (q_out // H if q_out and H else None))
        f["gated"] = bool(q_out and H and hd and q_out == 2 * H * hd)
        f["dh"] = hd
        k = get("k_proj.weight", "k_proj.linear.weight")
        f["Hkv"] = (k[0] // hd) if (k and hd) else (cget(cfg, "num_key_value_heads", None) or H)
        if qkv:
            f["Hkv"] = H
    if f["mixer"] == "gdn":
        f["gdn"] = {k: cget(cfg, k, None) for k in ("linear_num_key_heads", "linear_num_value_heads",
                                                      "linear_key_head_dim", "linear_value_head_dim",
                                                      "linear_conv_kernel_dim")}
    up = get("mlp.up_proj.weight", "mlp.up_proj.linear.weight", "mlp.fc1.weight", "mlp.linear_fc1.weight",
             "ffw_layer_1.linear.weight")
    gu = get("experts.gate_up_proj")
    f["dff"] = up[0] if up else (gu[-1] // 2 if gu else None)
    f["E"] = gu[0] if gu else None
    f["topk"] = cget(cfg, "num_experts_per_tok", None)
    f["ple_dim"] = get("per_layer_input_gate.weight")[0] if f["ple"] else None
    return f


def mixer_lines(f):
    """Equations for the token mixer, as (lhs, rhs, comment) triples."""
    L = []
    if f["mixer"] == "gdn":
        g = f["gdn"]
        L += [(r"[q,\,k,\,v]", r"\op{SiLU}\big(\op{CausalConv1d}(W_{qkv}\,u)\big)",
               f"short causal conv (kernel {g['linear_conv_kernel_dim']}) over time"),
              (r"z", r"W_{z}\,u", "output gate"),
              (r"\beta_t", r"\sigma(W_{b}\,u_t)", "write strength in (0,1)"),
              (r"g_t", r"-e^{A_{\log}}\cdot\op{softplus}(W_{a}\,u_t + b_{dt})", "decay ($\\le 0$): how fast memory fades"),
              (r"q_t,\,k_t", r"\op{L2Norm}(q_t),\ \op{L2Norm}(k_t)", ""),
              (r"S_t", r"e^{g_t}\,S_{t-1}\big(I - \beta_t k_t k_t^{\top}\big) + \beta_t\, v_t k_t^{\top}",
               "fixed-size memory per head (delta rule)"),
              (r"o_t", r"S_t\, q_t", "read memory with the query"),
              (r"\op{Mixer}(u)", r"W_{\mathrm{out}}\big(\op{RMSNorm}(o)\odot\op{SiLU}(z)\big)", "")]
        return L
    b = lambda n: f" + b_{{{n}}}" if f["bias"] else ""
    if f["fused_qkv"]:
        L.append((r"[Q,\,K,\,V]", r"W_{qkv}\,u" + (r" + b_{qkv}" if f["bias"] else ""), "one fused projection"))
    else:
        if f["gated"]:
            L.append((r"[Q,\,G]", r"W_Q\,u" + b("Q"), "query and an output gate"))
        else:
            L.append((r"Q", r"W_Q\,u" + b("Q"), f"$H={f['H']}$ heads of size $d_h={f['dh']}$"))
        if f["kv_shared"]:
            L.append((r"K,\,V", r"K^{(\ell^*)},\,V^{(\ell^*)}",
                      "from last non-shared layer $\\ell^*$ (no K/V weights)"))
        else:
            L.append((r"K,\ V", r"W_K\,u" + b("K") + r",\quad W_V\,u" + b("V"), f"$H_{{kv}}={f['Hkv']}$ key/value heads"))
    if f["qk_norm"]:
        L.append((r"Q,\ K", r"\op{RMSNorm}_q(Q),\ \op{RMSNorm}_k(K)" if not f["kv_shared"] else r"\op{RMSNorm}_q(Q),\ K",
                  "per-head normalisation"))
    if f["v_norm"] and not f["kv_shared"]:
        L.append((r"V", r"\op{RMSNorm}(V)", "no learned scale"))
    if f["rope"]:
        L.append((r"Q,\ K", r"\op{RoPE}(Q),\ \op{RoPE}(K)" if f["role"] == "text" else r"\op{RoPE}_{2D}(Q),\ \op{RoPE}_{2D}(K)",
                  "rotate by position" if f["role"] == "text" else "rotate by (row, column) patch position"))
    mask = ("causal + sliding window of $w=" + str(f["window"]) + "$" if f["sliding"] else
            "causal" if f["causal"] else "bidirectional")
    grp = r"_{g(h)}" if f["Hkv"] and f["Hkv"] != f["H"] else r"_h"
    L.append((r"S_h", r"s\, Q_h K" + grp + r"^{\top} + M", f"mask $M$: {mask}"))
    if f["sinks"]:
        L.append((r"P_h", r"\op{softmax}\big([\,S_h \;\|\; z_h\,]\big)_{1:T}", "learned sink logit $z_h$ soaks up attention, then is dropped"))
    else:
        L.append((r"P_h", r"\op{softmax}(S_h)", "attention weights"))
    L.append((r"O_h", r"P_h\, V" + grp, ""))
    out = r"W_O\,[O_1;\dots;O_H]" + (r" + b_O" if f["o_bias"] else "")
    if f["gated"]:
        out = r"W_O\big([O_1;\dots;O_H]\odot\sigma(G)\big)" + (r" + b_O" if f["o_bias"] else "")
    L.append((r"\op{Attn}(u)", out, ""))
    return L


def ffn_lines(f):
    a = f["act"]
    if f["ffn"] == "moe":
        return [(r"\lambda", r"W_r\,u + b_r \in \mathbb{R}^{" + str(f["E"]) + "}", "router score per expert"),
                (r"\mathcal{I}", r"\op{TopK}(\lambda,\ " + str(f["topk"]) + ")", f"pick {f['topk']} of {f['E']} experts"),
                (r"w", r"\op{softmax}(\lambda_{\mathcal{I}})", "mixing weights"),
                (r"[a,\,c]", r"W^{gu}_e\,u + b^{gu}_e", "per expert $e$: gate $a$, up $c$"),
                (r"\op{E}_e(u)", r"W^{d}_e\big((c+1)\odot a\,\sigma(1.702\,a)\big) + b^{d}_e", "clamped SwiGLU expert"),
                (r"\op{FFN}(u)", r"\textstyle\sum_{e\in\mathcal{I}} w_e\,\op{E}_e(u)", "")]
    if f["ffn"] == "gated":
        return [(r"\op{FFN}(u)", r"W_{\mathrm{down}}\big(" + a + r"(W_{\mathrm{gate}}\,u)\odot W_{\mathrm{up}}\,u\big)",
                 f"gated MLP, width {f['dff']}")]
    if f["ffn"] == "plain":
        return [(r"\op{FFN}(u)", r"W_2\," + a + r"(W_1 u + b_1) + b_2", f"MLP, width {f['dff']}")]
    return []


def wiring_lines(f):
    """How the block's sub-layers and residual connections are wired."""
    M = r"\op{Mixer}" if f["mixer"] == "gdn" else r"\op{Attn}"
    if f["conformer"]:
        return [(r"h_1", r"h + \tfrac12\,\op{FFN}_1(h)", "macaron half-step FFN"),
                (r"h_2", r"h_1 + \op{RMSNorm}_{\mathrm{post}}\big(\op{Attn}_{rel}(\op{RMSNorm}_{\mathrm{pre}}(h_1))\big)",
                 "local attention, relative positions"),
                (r"h_3", r"h_2 + W_{\mathrm{end}}\,\op{SiLU}\big(\op{RMSNorm}(\op{DWConv}(\op{GLU}(W_{\mathrm{start}}\op{RMSNorm}(h_2))))\big)",
                 "convolution module"),
                (r"h_4", r"h_3 + \tfrac12\,\op{FFN}_2(h_3)", ""),
                (r"\mathcal{B}(h)", r"\op{RMSNorm}_{\mathrm{out}}(h_4)", ""),
                (r"\op{FFN}_i(u)", r"\op{RMSNorm}\big(W_2\,\op{SiLU}(W_1\,\op{RMSNorm}(u))\big)", "")]
    N = r"\op{LN}" if f["ln"] else r"\op{RMSNorm}"
    if f["sandwich"]:
        L = [(r"a", r"h + " + N + r"_{\mathrm{pa}}\big(" + M + r"(" + N + r"_{\mathrm{in}}(h))\big)", "norm before AND after attention"),
             (r"b", r"a + " + N + r"_{\mathrm{pf}}\big(\op{FFN}(" + N + r"_{\mathrm{pre}}(a))\big)", "norm before AND after the MLP")]
        last = "b"
        if f["ple"]:
            L.append((r"c", r"b + \op{PLE}_\ell(b)", f"inject this layer's {f['ple_dim']}-dim per-layer embedding $p_\\ell$"))
            L.append((r"\op{PLE}_\ell(u)", r"\op{RMSNorm}_{\mathrm{ple}}\big(W^{\mathrm{ple}}_{\mathrm{proj}}(" + f["act"] +
                      r"(W^{\mathrm{ple}}_{\mathrm{gate}}\,u)\odot p_\ell)\big)", ""))
            last = "c"
        if f["layer_scalar"]:
            L.append((r"\mathcal{B}(h)", r"\gamma_\ell\cdot " + last, "fixed per-layer scalar"))
        elif last == "b":
            L[-1] = (r"\mathcal{B}(h)",) + L[-1][1:]
        else:
            L.append((r"\mathcal{B}(h)", last, ""))
        return L
    return [(r"a", r"h + " + M + r"(" + N + r"_{1}(h))", "pre-norm + residual"),
            (r"\mathcal{B}(h)", r"a + \op{FFN}(" + N + r"_{2}(a))", "pre-norm + residual")]


def align(lines, op="="):
    out = [r"\begin{align*}"]
    for i, (l, r, c) in enumerate(lines):
        end = r" \\" if i < len(lines) - 1 else ""
        out.append(f"  {l} &{op} {r}" + (r" && \text{\footnotesize " + c + "}" if c else "") + end)
    out.append(r"\end{align*}")
    return out


def block_algorithm(f, code, name, d_sym):
    """Step-by-step pseudocode of one block, with tensor shapes."""
    S = []
    st = lambda x, c="": S.append(r"\State $" + x + "$" + (r" \Comment{" + c + "}" if c else ""))
    sub = lambda t: S.append(r"\Statex \hspace{-1.2em}\textit{" + t + "}")
    BT = r"B{\times}T"
    if f["conformer"]:
        st(r"h \gets h + \tfrac12\,\op{FFN}_1(h)", "first half-step feed-forward")
        st(r"h \gets h + \op{RMSNorm}(\op{Attn}_{rel}(\op{RMSNorm}(h)))", "local self-attention over audio frames")
        st(r"h \gets h + \op{ConvModule}(h)", "GLU $\\to$ depthwise conv $\\to$ norm $\\to$ SiLU $\\to$ linear")
        st(r"h \gets h + \tfrac12\,\op{FFN}_2(h)", "second half-step feed-forward")
        st(r"h \gets \op{RMSNorm}(h)", "")
        S.append(r"\State \Return $h$")
        return S
    N = r"\op{LN}" if f["ln"] else r"\op{RMSNorm}"
    st(r"r \gets h", "keep the input for the residual (skip) connection")
    st(r"u \gets " + N + r"(h)", "normalise every token vector")
    if f["mixer"] == "gdn":
        sub("token mixing: Gated DeltaNet (linear attention, fixed-size memory)")
        g = f["gdn"]
        st(r"q,k,v \gets \op{split}(\op{SiLU}(\op{CausalConv1d}(u W_{qkv}^{\top})))",
           f"{g['linear_num_key_heads']} q/k heads of {g['linear_key_head_dim']}, {g['linear_num_value_heads']} v heads of {g['linear_value_head_dim']}")
        st(r"z \gets u W_z^{\top};\ \beta \gets \sigma(u W_b^{\top});\ g \gets -e^{A_{\log}}\op{softplus}(u W_a^{\top}+b_{dt})", "gate, write strength, decay")
        S.append(r"\For{$t \gets 1$ \textbf{to} $T$} \Comment{conceptually sequential; computed in chunks}")
        st(r"S \gets e^{g_t} S\,(I-\beta_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}", "forget a bit, overwrite what $k_t$ points at")
        st(r"o_t \gets S\, q_t", "read")
        S.append(r"\EndFor")
        st(r"m \gets \op{RMSNorm}(o)\odot \op{SiLU}(z)\ W_{\mathrm{out}}^{\top}", f"back to ${BT}{{\\times}}{f['d']}$")
    elif f["mixer"] == "attn":
        sub("token mixing: multi-head self-attention")
        H, Hkv, dh = f["H"], f["Hkv"], f["dh"]
        if f["fused_qkv"]:
            st(r"Q,K,V \gets \op{split}(u W_{qkv}^{\top}" + (" + b" if f["bias"] else "") + ")", f"each ${BT}{{\\times}}{H}{{\\times}}{dh}$")
        else:
            st(r"Q \gets u W_Q^{\top}" + (" + b_Q" if f["bias"] else ""), f"${BT}{{\\times}}{H}{{\\times}}{dh}$ ({H} query heads)"
               + ("; second half is a gate $G$" if f["gated"] else ""))
            if f["kv_shared"]:
                st(r"K, V \gets K^{(\ell^*)}, V^{(\ell^*)}", "re-use keys/values of an earlier layer (saves compute and KV-cache)")
            else:
                st(r"K \gets u W_K^{\top}" + (" + b_K" if f["bias"] else "") + r";\ V \gets u W_V^{\top}" + (" + b_V" if f["bias"] else ""),
                   f"${BT}{{\\times}}{Hkv}{{\\times}}{dh}$ ({Hkv} key/value heads" + (f", each shared by {H // Hkv} query heads)" if Hkv and H != Hkv else ")"))
        if f["qk_norm"]:
            st(r"Q \gets \op{RMSNorm}(Q)" + ("" if f["kv_shared"] else r";\ K \gets \op{RMSNorm}(K)"), "normalise each head vector (stabilises training)")
        if f["v_norm"] and not f["kv_shared"]:
            st(r"V \gets \op{RMSNorm}(V)", "")
        if f["rope"]:
            st(r"Q, K \gets \op{RoPE}(Q, K)", "encode position by rotating vector pairs" if f["role"] == "text" else "2-D rotary position (row, column)")
        mask = (f"causal + only the last {f['window']} tokens" if f["sliding"] else
                "causal: a token sees only earlier tokens" if f["causal"] else "every patch/frame sees every other")
        st(r"P \gets \op{softmax}(s\, Q K^{\top} + M)", "mask: " + mask + ("; plus a learned sink column" if f["sinks"] else ""))
        st(r"O \gets P V", f"${BT}{{\\times}}{H}{{\\times}}{dh}$")
        if f["gated"]:
            st(r"O \gets O \odot \sigma(G)", "output gate decides how much of each head to keep")
        st(r"m \gets O\, W_O^{\top}" + (" + b_O" if f["o_bias"] else ""), f"concatenate heads, project back to ${BT}{{\\times}}{f['d']}$")
    if f["sandwich"]:
        st(r"h \gets r + \op{RMSNorm}(m)", "post-attention norm, then add residual")
    else:
        st(r"h \gets r + m", "add residual")
    sub("channel mixing: feed-forward network (each token independently)")
    st(r"r \gets h;\ u \gets " + N + r"(h)", "")
    if f["ffn"] == "moe":
        st(r"\lambda \gets u W_r^{\top} + b_r", f"${BT}{{\\times}}{f['E']}$ router scores")
        st(r"\mathcal{I}, w \gets \op{TopK}(\lambda, {k}),\ \op{softmax}".replace("{k}", str(f["topk"])), f"each token picks {f['topk']} of {f['E']} experts")
        st(r"m \gets \sum_{e\in\mathcal{I}} w_e\,\op{E}_e(u)", f"expert: clamped SwiGLU, width {f['dff']}")
    elif f["ffn"] == "gated":
        st(r"m \gets \big(" + f["act"] + r"(u W_{\mathrm{gate}}^{\top}) \odot u W_{\mathrm{up}}^{\top}\big) W_{\mathrm{down}}^{\top}",
           f"expand to {f['dff']}, gate, shrink back to {f['d']}")
    elif f["ffn"] == "plain":
        st(r"m \gets " + f["act"] + r"(u W_1^{\top} + b_1) W_2^{\top} + b_2", f"expand to {f['dff']}, shrink back to {f['d']}")
    if f["sandwich"]:
        st(r"h \gets r + \op{RMSNorm}(m)", "post-feed-forward norm, then add residual")
    else:
        st(r"h \gets r + m", "add residual")
    if f["ple"]:
        sub("per-layer embedding injection")
        st(r"h \gets h + \op{RMSNorm}\big((" + f["act"] + r"(h W^{\mathrm{ple}\top}_{\mathrm{gate}}) \odot p_\ell) W^{\mathrm{ple}\top}_{\mathrm{proj}}\big)",
           f"gate down to {f['ple_dim']}, multiply by $p_\\ell$, project back up")
    if f["layer_scalar"]:
        st(r"h \gets \gamma_\ell\, h", "fixed per-layer scale")
    S.append(r"\State \Return $h$")
    return S


# ---------------------------------------------------------------------------
# Glossary: plain-English description of every kind of layer/weight
# ---------------------------------------------------------------------------

GLOSSARY = [
    # Gemma-4 audio tower
    (r"audio_tower\.subsample_conv_projection\.layer\d\.conv", "Subsampling convolution",
     "2-D convolution (stride 2) over the time $\\times$ mel-frequency spectrogram. Two of them shorten the time axis 4$\\times$, so the encoder works on fewer, richer frames."),
    (r"audio_tower\.subsample_conv_projection\.layer\d\.norm", "Subsampling norm", "LayerNorm over channels after each subsampling convolution."),
    (r"audio_tower\.subsample_conv_projection\.input_proj_linear", "Audio input projection",
     "Flattens each frame's conv features and projects them to the audio-encoder width."),
    (r"feed_forward[12]\.pre_layer_norm", "Conformer FFN pre-norm", "RMSNorm before each of the two conformer feed-forward modules."),
    (r"feed_forward[12]\.ffw_layer_1", "Conformer FFN up-projection", "Expands each frame vector (e.g. 1024$\\to$4096) before the SiLU non-linearity."),
    (r"feed_forward[12]\.ffw_layer_2", "Conformer FFN down-projection",
     "Projects back to encoder width. The result is halved (``macaron'' half-step) and added to the residual."),
    (r"feed_forward[12]\.post_layer_norm", "Conformer FFN post-norm", "RMSNorm on the feed-forward output before the residual add."),
    (r"audio_tower.*norm_pre_attn", "Pre-attention norm", "RMSNorm applied to frames before audio self-attention."),
    (r"audio_tower.*self_attn\.(q|k|v)_proj", "Audio Q/K/V projections",
     "Turn each audio frame into a query (what am I looking for), key (what do I contain) and value (what do I pass on). ``Clippable'' linears clamp activations for numerical safety."),
    (r"audio_tower.*self_attn\.relative_k_proj", "Relative-position key projection",
     "Projects relative-distance encodings so attention scores depend on how far apart two frames are (Transformer-XL style) instead of RoPE."),
    (r"audio_tower.*self_attn\.per_dim_scale", "Per-dimension query scale", "A learned scale per head dimension of the query, replacing the fixed $1/\\sqrt{d_h}$."),
    (r"audio_tower.*self_attn\.post", "Audio attention output projection", "Mixes the concatenated head outputs back to encoder width."),
    (r"audio_tower.*norm_post_attn", "Post-attention norm", "RMSNorm on the attention output before the residual add."),
    (r"lconv1d\.pre_layer_norm", "Conv-module pre-norm", "RMSNorm before the light-weight convolution module."),
    (r"lconv1d\.linear_start", "Conv-module input projection (GLU)",
     "Doubles the width; a Gated Linear Unit then uses one half of the channels to gate the other half."),
    (r"lconv1d\.depthwise_conv1d", "Depthwise causal 1-D convolution",
     "Each channel is convolved over time with its own small kernel (5 frames), picking up local acoustic patterns such as phoneme transitions."),
    (r"lconv1d\.conv_norm", "Conv-module norm", "RMSNorm after the depthwise convolution."),
    (r"lconv1d\.linear_end", "Conv-module output projection", "Projects the convolved features back before the residual add."),
    (r"audio_tower.*norm_out", "Block output norm", "RMSNorm at the end of every conformer block."),
    (r"audio_tower\.output_proj", "Audio output projection", "Maps the last audio-encoder states to the size the audio embedder expects."),
    (r"embed_audio\.embedding_projection", "Audio-to-text projector",
     "Normalises audio-encoder outputs and projects them into the language model's hidden size, so they can take the place of text-token embeddings."),
    (r"embed_vision\.embedding_projection", "Image-to-text projector",
     "Normalises pooled vision features and projects them into the language model's hidden size (``soft image tokens'')."),
    # Gemma-4 vision tower
    (r"vision_tower\.patch_embedder\.input_proj", "Patch embedding",
     "Each image patch (16$\\times$16 pixels $\\times$ 3 colours, flattened) is mapped linearly to one vector."),
    (r"vision_tower\.patch_embedder\.position_embedding_table", "2-D position table",
     "Learned vectors for a patch's x and y coordinate, added so the encoder knows where each patch sits in the image."),
    # SigLIP (Gemma-3) vision tower
    (r"embeddings\.patch_embedding", "Patch embedding (Conv2d)",
     "A strided convolution that cuts the image into square patches (14$\\times$14 pixels) and maps each patch to one vector."),
    (r"embeddings\.position_embedding", "Position embedding", "One learned vector per patch position, added so the encoder knows where each patch is."),
    (r"layer_norm1", "Pre-attention LayerNorm", "LayerNorm (mean/variance normalisation with scale and bias) before self-attention."),
    (r"layer_norm2", "Pre-MLP LayerNorm", "LayerNorm before the MLP."),
    (r"post_layernorm", "Final vision LayerNorm", "Normalises the vision encoder's output patches."),
    (r"multi_modal_projector\.mm_soft_emb_norm", "Projector norm",
     "RMSNorm on image features after average-pooling the 4096 patches down to 256 image tokens."),
    (r"multi_modal_projector\.mm_input_projection_weight", "Image-to-text projection",
     "Matrix mapping each pooled image feature (vision width) into the language model's hidden size."),
    # Qwen3.5 vision tower
    (r"visual\.patch_embed\.proj", "3-D patch embedding (Conv3d)",
     "Cuts images/videos into 2-frame $\\times$ 16 $\\times$ 16-pixel cubes and maps each cube to one vector."),
    (r"visual\.pos_embed", "Position embedding", "Learned position vectors, interpolated to the image's patch grid."),
    (r"visual\.blocks\.N\.norm1", "Pre-attention LayerNorm", "LayerNorm before vision self-attention."),
    (r"visual\.blocks\.N\.norm2", "Pre-MLP LayerNorm", "LayerNorm before the vision MLP."),
    (r"visual\.merger\.norm", "Merger LayerNorm", "Normalises patch features before 2$\\times$2 neighbouring patches are merged."),
    (r"visual\.merger\.linear_fc1", "Merger MLP layer 1",
     "Takes 4 neighbouring patches concatenated (2$\\times$2 merge, so 4$\\times$ fewer image tokens) and mixes them."),
    (r"visual\.merger\.linear_fc2", "Merger MLP layer 2 (to text width)", "Projects merged patches into the language model's hidden size."),
    (r"attn\.qkv", "Fused Q/K/V projection", "One matrix that produces queries, keys and values at once (split afterwards)."),
    (r"attn\.proj$|attn\.proj\.", "Attention output projection", "Mixes the concatenated head outputs back to the model width."),
    (r"self_attn\.out_proj", "Attention output projection", "Mixes the concatenated head outputs back to the model width."),
    (r"mlp\.(fc1|linear_fc1)", "MLP up-projection (fc1)", "Expands each vector to the wider MLP size before the GELU non-linearity."),
    (r"mlp\.(fc2|linear_fc2)", "MLP down-projection (fc2)", "Shrinks back to the model width."),
    # Gemma-4 per-layer embeddings (PLE)
    (r"embed_tokens_per_layer", "Per-layer token embedding (PLE)",
     "A second embedding table looked up from the same token ids. It holds one small vector per token for every decoder layer, which gives each layer extra token-specific information cheaply. It can sit in CPU/flash memory, which is what the ``E'' (effective size) in the model name refers to."),
    (r"per_layer_model_projection", "PLE context projection",
     "Projects the main token embedding into one small vector per layer. Added to the PLE lookup, so the per-layer input depends on context as well as token identity."),
    (r"per_layer_projection_norm", "PLE projection norm", "RMSNorm on the projected per-layer inputs."),
    (r"per_layer_input_gate", "PLE gate (down-projection)", "Inside each layer: projects the hidden state down to the PLE size; after GELU it multiplies this layer's per-layer embedding $p_\\ell$."),
    (r"per_layer_projection\.", "PLE up-projection", "Projects the gated per-layer signal back up to the hidden size before it is added to the residual stream."),
    (r"post_per_layer_input_norm", "PLE output norm", "RMSNorm on the per-layer contribution before the residual add."),
    # Text embeddings / head
    (r"embed_tokens", "Token embedding",
     "Lookup table with one learned vector per vocabulary token. Turns token ids into the vectors the network computes with (Gemma multiplies them by $\\sqrt{d}$)."),
    (r"^lm_head", "LM head (output projection)",
     "Maps each final hidden vector to one score (logit) per vocabulary token. Softmax of the logits gives the next-token probabilities. When tied, it reuses the embedding matrix."),
    # Softmax attention
    (r"self_attn\.q_proj", "Query projection (q\\_proj)",
     "Turns each token vector into query vectors, one per head: ``what am I looking for?''."),
    (r"self_attn\.k_proj", "Key projection (k\\_proj)",
     "Turns each token into key vectors: ``what do I contain?''. Queries are compared with keys to decide where to attend. With grouped-query attention there are fewer key heads than query heads."),
    (r"self_attn\.v_proj", "Value projection (v\\_proj)", "Turns each token into value vectors: the information actually copied to tokens that attend to it."),
    (r"self_attn\.o_proj", "Output projection (o\\_proj)", "Concatenates all heads' results and mixes them back into one hidden-size vector per token."),
    (r"self_attn\.q_norm", "Query norm (q\\_norm)", "RMSNorm applied to every query head vector before RoPE. Keeps attention logits in a sane range (QK-norm)."),
    (r"self_attn\.k_norm", "Key norm (k\\_norm)", "RMSNorm applied to every key head vector before RoPE."),
    (r"self_attn\.sinks", "Attention sinks", "One learned logit per head that joins the softmax as an extra ``attend to nothing'' slot. Lets a head put weight nowhere instead of being forced onto real tokens."),
    # Gated DeltaNet linear attention
    (r"linear_attn\.in_proj_qkv", "DeltaNet Q/K/V projection",
     "Produces queries, keys and values for the linear-attention layer in one matrix."),
    (r"linear_attn\.in_proj_z", "DeltaNet output-gate projection", "Produces gate $z$. The normalised output is multiplied by SiLU($z$)."),
    (r"linear_attn\.in_proj_b", "DeltaNet write-strength projection", "Produces $\\beta=\\sigma(\\cdot)$ per head: how strongly the current token overwrites the memory."),
    (r"linear_attn\.in_proj_a", "DeltaNet decay projection", "Produces the input-dependent forgetting rate (with A\\_log and dt\\_bias)."),
    (r"linear_attn\.conv1d", "Short causal convolution",
     "Depthwise 1-D convolution over the last 4 tokens of Q/K/V. Gives cheap local context before the recurrent memory update."),
    (r"linear_attn\.A_log", "A\\_log (decay base)", "Per-head learned log decay rate. $-e^{A_{\\log}}$ sets how quickly old memory fades."),
    (r"linear_attn\.dt_bias", "dt\\_bias (decay bias)", "Per-head bias added before the softplus that controls the decay step."),
    (r"linear_attn\.norm", "Gated output RMSNorm", "Normalises each head's memory read-out and multiplies it by SiLU($z$)."),
    (r"linear_attn\.out_proj", "DeltaNet output projection", "Mixes the heads back into one hidden-size vector."),
    # MoE
    (r"mlp\.router", "MoE router", "A small linear layer that scores every expert for every token. The top-$k$ experts are run and their outputs mixed with softmax weights."),
    (r"experts\.gate_up_proj", "Expert gate+up projection (fused)",
     "For each expert: one matrix producing both the gate and the up-projection (interleaved), plus a bias. Every expert is its own small MLP."),
    (r"experts\.down_proj", "Expert down projection", "For each expert: projects the gated hidden activations back to the model width (plus bias)."),
    # Gated MLP
    (r"mlp\.gate_proj", "Gate projection (gate\\_proj)",
     "Expands the token vector to the MLP width and passes it through SiLU/GELU. The result acts as a soft on/off switch for each hidden unit."),
    (r"mlp\.up_proj", "Up projection (up\\_proj)",
     "Expands the token vector to the MLP width (the ``content'' path). It is multiplied element-wise by the gate (SwiGLU/GeGLU)."),
    (r"mlp\.down_proj", "Down projection (down\\_proj)", "Shrinks the gated hidden vector back to the model width so it can be added to the residual stream."),
    # Norms
    (r"input_layernorm", "Input norm (input\\_layernorm)",
     "RMSNorm applied to the residual stream before attention. Rescales each token vector to unit RMS, then multiplies by a learned per-channel weight."),
    (r"post_attention_layernorm", "post\\_attention\\_layernorm", None),
    (r"pre_feedforward_layernorm", "Pre-MLP norm (pre\\_feedforward\\_layernorm)", "RMSNorm on the residual stream before the MLP (Gemma)."),
    (r"post_feedforward_layernorm", "Post-MLP norm (post\\_feedforward\\_layernorm)", "RMSNorm on the MLP output before it is added back (Gemma ``sandwich'' norm)."),
    (r"(^|\.)norm\.weight$", "Final norm", "RMSNorm applied once after the last decoder layer, just before the LM head."),
]

POST_ATTN = {
    True: "RMSNorm applied to the attention output before it is added to the residual stream (Gemma ``sandwich'' norm).",
    False: "Despite the name, this is the pre-MLP norm: RMSNorm on the residual stream after attention, feeding the MLP.",
}

CKPT_ONLY_DESC = {
    "mtp": ("Multi-token-prediction (MTP) head",
            "An extra one-layer transformer that predicts the token after next. It concatenates RMSNorm(embedding of token $t{+}1$) "
            "with RMSNorm(final hidden state at $t$), projects $2d\\to d$ (\\texttt{fc}), runs one full-attention decoder layer, "
            "then a norm and the shared LM head. Serving engines (vLLM/SGLang) use it for speculative decoding. "
            "The transformers class does not instantiate it, so it is not counted in the total above."),
}


GLOSS_GROUPS = ["Language model", "Vision encoder", "Audio encoder", "Multimodal projector"]
TEXT_ORDER = ["embed", "PLE", "Query", "Key", "Value", "Output projection", "norm (q", "norm (k", "sinks",
              "DeltaNet", "convolution", "A\\_log", "dt\\_bias", "Gated output", "router", "Expert",
              "Gate", "Up", "Down", "Input norm", "post\\_attention", "Pre-MLP", "Post-MLP", "Final norm", "LM head"]


def gloss_group(path):
    c = component_of(path)
    return c if c in GLOSS_GROUPS else "Language model"


def gloss_rank(title):
    return next((i for i, k in enumerate(TEXT_ORDER) if k.lower() in title.lower()), len(TEXT_ORDER))


def glossary_for(recs, sandwich, gated):
    rows = OrderedDict()
    unmatched = []
    for r in recs:
        if r["tied_to"]:
            continue
        p = norm_path(r["path"])
        for pat, title, desc in GLOSSARY:
            if re.search(pat, p):
                if desc is None:
                    desc = POST_ATTN[sandwich]
                mod = norm_path(r["module"]) or r["path"]
                short = ".".join(x for x in mod.split(".") if x not in ("model", "N", "language_model", "linear"))
                short = ".".join(short.split(".")[-2:]) if short.count(".") > 1 else short
                if gated and title.startswith("Query projection"):
                    desc += " Here it also outputs an equally sized gate for the attention output (hence the doubled width)."
                key = (gloss_group(r["path"]), title)
                rows.setdefault(key, [desc, []])
                if short not in rows[key][1]:
                    rows[key][1].append(short)
                break
        else:
            unmatched.append(r["path"])
    if unmatched:
        print("    [glossary] no description for:", sorted({norm_path(u) for u in unmatched}))
    return OrderedDict(sorted(rows.items(), key=lambda kv: (GLOSS_GROUPS.index(kv[0][0]),
                                                          gloss_rank(kv[0][1]) if kv[0][0] == "Language model" else 0)))


# ---------------------------------------------------------------------------
# Parameter bookkeeping
# ---------------------------------------------------------------------------

def component_of(path):
    if "vision_tower" in path or path.startswith("model.visual"):
        return "Vision encoder"
    if "audio_tower" in path:
        return "Audio encoder"
    if "multi_modal_projector" in path or "embed_vision" in path or "embed_audio" in path:
        return "Multimodal projector"
    if "per_layer" in path:
        return "Per-layer embeddings (PLE)"
    if "embed_tokens" in path:
        return "Token embedding"
    if path.startswith("lm_head"):
        return "LM head"
    if "self_attn" in path or "linear_attn" in path:
        return "Attention / token mixer"
    if ".mlp." in path:
        return "MLP / MoE"
    if "norm" in path:
        return "Norms"
    return "Other"


# ---------------------------------------------------------------------------
# LaTeX document
# ---------------------------------------------------------------------------

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{geometry}
\geometry{margin=0.75in}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage[title]{appendix}
\newcommand{\op}[1]{\operatorname{#1}}
\algrenewcommand\algorithmiccomment[1]{\hfill{\color{gray!70!black}\footnotesize$\triangleright$ #1}}
\setlength{\LTpre}{4pt}\setlength{\LTpost}{4pt}
\allowdisplaybreaks
\theoremstyle{definition}
\newtheorem{definition}{Definition}[section]
"""


def table(spec, header, rows, caption=None):
    t = [r"\begin{longtable}{" + spec + "}"]
    if caption:
        t.append(r"\caption{" + caption + r"}\\")
    t += [r"\toprule", header + r" \\", r"\midrule", r"\endfirsthead",
          r"\toprule", header + r" \\", r"\midrule", r"\endhead",
          r"\midrule \multicolumn{" + str(ncols(spec)) + r"}{r}{\textit{continued on next page}} \\", r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    t += [r + r" \\" for r in rows]
    t.append(r"\end{longtable}")
    return t


def ncols(spec):
    return len(re.findall(r"[lrc]|p\{[^}]*\}|X", re.sub(r">\{[^}]*\}", "", spec)))



def hyper_rows(config):
    t = config.get_text_config()
    keys = [("hidden_size", "Hidden size $d$"), ("num_hidden_layers", "Decoder layers $L$"),
            ("num_attention_heads", "Query heads $H$"), ("num_key_value_heads", "Key/value heads $H_{kv}$"),
            ("head_dim", "Head dim $d_h$"), ("global_head_dim", "Head dim (global layers)"),
            ("intermediate_size", "MLP width $d_{ff}$"), ("num_local_experts", "Experts $E$"),
            ("num_experts_per_tok", "Active experts per token"),
            ("linear_num_key_heads", "DeltaNet q/k heads"), ("linear_num_value_heads", "DeltaNet v heads"),
            ("linear_key_head_dim", "DeltaNet q/k head dim"), ("linear_value_head_dim", "DeltaNet v head dim"),
            ("linear_conv_kernel_dim", "DeltaNet conv kernel"),
            ("hidden_size_per_layer_input", "PLE dim per layer"), ("num_kv_shared_layers", "KV-shared layers"),
            ("vocab_size", "Vocabulary $V$"), ("max_position_embeddings", "Max context length"),
            ("sliding_window", "Sliding window $w$"), ("final_logit_softcapping", "Logit soft-cap"),
            ("rms_norm_eps", "RMSNorm $\\epsilon$"), ("tie_word_embeddings", "Tied input/output embeddings")]
    rows = []
    for k, lab in keys:
        v = cget(t, k, None)
        if v is None and k == "tie_word_embeddings":
            v = cget(config, k, None)
        if v is None or (k == "sliding_window" and not any(
                x == "sliding_attention" for x in (cget(t, "layer_types", None) or []))):
            continue
        rows.append(f"{lab} & {fmt(v) if isinstance(v, int) and not isinstance(v, bool) else esc(v)}")
    rows.append(f"Activation & ${act_of(t)}$")
    rp = cget(t, "rope_parameters", None) or {}
    theta = cget(t, "rope_theta", None) or (rp.get("rope_theta") if isinstance(rp, dict) else None)
    if theta:
        rows.append(f"RoPE base $\\theta$ & {fmt(int(theta))}")
    return rows


def arch_signature(cls, recs):
    return cls, [(r["path"], tuple(r["shape"]), r["dtype"], r["tied_to"]) for r in recs]


# ---------------------------------------------------------------------------
# Mathematical representation (top-down formal treatment)
# ---------------------------------------------------------------------------

# Weight name suffix -> math symbol. First match wins, so specific names come first.
WSYM = [
    (("linear_attn.in_proj_qkv.weight", "attn.qkv.weight"), r"W_{qkv}"),
    (("in_proj_z.weight",), r"W_z"), (("in_proj_a.weight",), r"W_a"), (("in_proj_b.weight",), r"W_b"),
    (("linear_attn.out_proj.weight",), r"W_{\mathrm{out}}"),
    (("linear_attn.conv1d.weight",), r"K_{\mathrm{conv}}"),
    (("relative_k_proj.weight",), r"W_{\mathrm{rel}}"),
    (("q_proj.weight", "q_proj.linear.weight"), r"W_Q"),
    (("k_proj.weight", "k_proj.linear.weight"), r"W_K"),
    (("v_proj.weight", "v_proj.linear.weight"), r"W_V"),
    (("o_proj.weight", "o_proj.linear.weight", "attn.proj.weight", "self_attn.post.linear.weight",
      "self_attn.out_proj.weight"), r"W_O"),
    (("mlp.gate_proj.weight", "mlp.gate_proj.linear.weight"), r"W_{\mathrm{gate}}"),
    (("mlp.up_proj.weight", "mlp.up_proj.linear.weight"), r"W_{\mathrm{up}}"),
    (("mlp.down_proj.weight", "mlp.down_proj.linear.weight"), r"W_{\mathrm{down}}"),
    (("mlp.fc1.weight", "mlp.linear_fc1.weight"), r"W_1"), (("mlp.fc2.weight", "mlp.linear_fc2.weight"), r"W_2"),
    (("mlp.router.weight",), r"W_r"),
    (("experts.gate_up_proj",), r"W^{gu}_e"), (("experts.down_proj",), r"W^{d}_e"),
    (("per_layer_input_gate.weight",), r"W^{\mathrm{ple}}_{\mathrm{gate}}"),
    (("per_layer_projection.weight",), r"W^{\mathrm{ple}}_{\mathrm{proj}}"),
    (("feed_forward1.ffw_layer_1.linear.weight",), r"W_1^{[1]}"), (("feed_forward1.ffw_layer_2.linear.weight",), r"W_2^{[1]}"),
    (("feed_forward2.ffw_layer_1.linear.weight",), r"W_1^{[2]}"), (("feed_forward2.ffw_layer_2.linear.weight",), r"W_2^{[2]}"),
    (("lconv1d.linear_start.linear.weight",), r"W_{\mathrm{start}}"),
    (("lconv1d.linear_end.linear.weight",), r"W_{\mathrm{end}}"),
    (("lconv1d.depthwise_conv1d.weight",), r"K_{\mathrm{dw}}"),
]


def mnum(n):
    """Integer for math mode (thousands separators without extra space)."""
    return fmt(n).replace(",", "{,}")


def block_params(b):
    """Matrices of one block with their math symbols and shapes (out x in), plus the leftover count."""
    out, used = [], 0
    for r in b["recs"]:
        sym = next((s for sufs, s in WSYM if any(r["rel"].endswith(x) for x in sufs)), None)
        if sym is None:
            continue
        shp = list(r["shape"])
        if r["rel"].startswith("mlp.experts.") or ".experts." in r["rel"]:
            txt = sym + r"\in\mathbb{R}^{" + str(shp[2]) + r"\times" + str(shp[1]) + r"},\ e=1..{" + str(shp[0]) + "}"
        else:
            txt = sym + r"\in\mathbb{R}^{" + r"\times".join(str(x) for x in shp) + "}"
        out.append(txt)
        used += r["numel"]
    return out, b["params"] - used


def block_composition(f):
    """The block as a composition of residual sub-layers (id + g)."""
    I = r"\mathrm{id}"
    N = r"\op{LN}" if f["ln"] else r"\op{RMSNorm}"
    M = r"\op{Mixer}" if f["mixer"] == "gdn" else r"\op{Attn}"
    if f["conformer"]:
        g = [r"\tfrac12\op{FFN}_1", r"\op{RMSNorm}_{\mathrm{post}}\circ\op{Attn}_{rel}\circ\op{RMSNorm}_{\mathrm{pre}}",
             r"\op{Conv}", r"\tfrac12\op{FFN}_2"]
        return (r"\op{RMSNorm}_{\mathrm{out}}\circ" + r"\circ".join(f"({I}+{x})" for x in reversed(g))), g, ""
    if f["sandwich"]:
        g = [N + r"_{\mathrm{pa}}\circ " + M + r"\circ " + N + r"_{\mathrm{in}}",
             N + r"_{\mathrm{pf}}\circ\op{FFN}\circ " + N + r"_{\mathrm{pre}}"]
    else:
        g = [M + r"\circ " + N + r"_{1}", r"\op{FFN}\circ " + N + r"_{2}"]
    if f["ple"]:
        g.append(r"\op{PLE}_\ell")
    s = r"\circ".join(f"({I}+{x})" for x in reversed(g))
    scale = r"\gamma_\ell" if f["layer_scalar"] else ""
    if scale:
        s = scale + r"\cdot\big(" + s + r"\big)"
    return s, g, scale


def pattern_str(groups):
    """Layer-type sequence left to right, e.g. (S^4 F)^7."""
    parts = []
    for unit, k, _ in groups:
        u = r"\,".join(r"\mathrm{" + c + "}" + (f"^{{{r}}}" if r > 1 else "") for c, r in runs(unit))
        if k > 1:
            if len(runs(unit)) == 1 and runs(unit)[0][1] > 1:
                c, r = runs(unit)[0]
                u = r"\mathrm{" + c + "}" + f"^{{{r * k}}}"
            else:
                u = (r"(" + u + ")" if len(runs(unit)) > 1 else u) + f"^{{{k}}}"
        parts.append(u)
    return r"\,".join(parts)


def valset(vals):
    vals = sorted({v for v in vals if v is not None})
    if not vals:
        return None
    return str(vals[0]) if len(vals) == 1 else r"\{" + ",".join(str(v) for v in vals) + r"\}"


def BL(code, l):
    return r"\mathcal{B}^{(" + str(l) + r")}_{\mathrm{" + code + "}}"


def math_section(name_tex, stacks, text_stack, towers, d, V, tied, softcap, has_ple, gemma_scale, total):
    tex = []
    L = text_stack["n"]
    codes = text_stack["codes"]
    tblocks = text_stack["blocks"]
    Vs = mnum(V) if isinstance(V, int) else V
    f0 = tblocks[0]["f"]
    ple_arg = r",\,p_\ell" if has_ple else ""

    tex.append(r"\section{Mathematical Representation}\label{sec:func}")
    tex.append(r"We describe the model \emph{top-down}. First the whole network is written as one composition of "
               r"functions (Section~\ref{sec:comp}). Then every function that appears in that composition is defined "
               r"(Section~\ref{sec:layers}), down to individual neurons and attention heads. The remaining subsections "
               r"describe the parameters, the outputs, the intermediate states, and the local linear behaviour.")

    # 2.1 composition -------------------------------------------------------------
    tex.append(r"\subsection{The model as a composition of layers}\label{sec:comp}")
    first = L if L <= 6 else max(3, min(len(text_stack["groups"][0][0]) if text_stack["groups"][0][1] > 1 else 4, 6))
    if L <= 6:
        seq = [BL(codes[l - 1], l) for l in range(L, 0, -1)]
    else:
        seq = [BL(codes[L - 1], L), r"\cdots"] + [BL(codes[l - 1], l) for l in range(first, 0, -1)]
    inp = r"E_{\mathrm{mm}}" if towers else r"E"
    tex.append(r"\begin{definition}[" + name_tex + r" as an $L$-layer decoder]")
    tex.append(r"The model is a function $f_\theta:\mathcal{X}\to\mathcal{Y}$ with parameters $\theta$. It maps a "
               r"sequence of $T$ token ids $x\in\mathcal{X}=\{1,\dots,V\}^{T}$ to one logit vector per position, "
               r"$y\in\mathcal{Y}=\mathbb{R}^{T\times V}$ ($V=" + Vs + r"$). It is a composition of an embedding $" + inp +
               r"$, $L=" + str(L) + r"$ decoder blocks $\mathcal{B}^{(\ell)}$ and an output head $\Psi$:")
    tex.append(r"\begin{equation}")
    tex.append(r"  f_\theta \;=\; \Psi\circ\mathcal{B}^{(L)}\circ\mathcal{B}^{(L-1)}\circ\cdots\circ\mathcal{B}^{(2)}\circ\mathcal{B}^{(1)}\circ " + inp)
    tex.append(r"\end{equation}")
    tex.append(r"Every block maps $\mathbb{R}^{T\times d}\to\mathbb{R}^{T\times d}$ ($d=" + str(d) + r"$), so the blocks "
               r"can be stacked in any number. Block $\ell$ has its own parameters $\theta_\ell$ and is stored in the "
               r"checkpoint as \texttt{" + esc(text_stack["path"]) + r".}$(\ell-1)$." +
               (r" Each block also receives its own per-layer embedding $p_\ell$ (Definition~\ref{def:embed})." if has_ple else ""))
    tex.append(r"\end{definition}")

    ntypes = len(tblocks)
    tex.append(r"\begin{definition}[Block types]")
    if ntypes == 1:
        tex.append(r"All $L$ blocks have the same functional form $\mathcal{B}_{\mathrm{" + tblocks[0]["code"] + r"}}$ "
                   r"(" + esc(tblocks[0]["name"]) + r"). They differ only in their weights: "
                   r"$\mathcal{B}^{(\ell)}=\mathcal{B}_{\mathrm{" + tblocks[0]["code"] + r"}}(\,\cdot\,;\theta_\ell)$.")
    else:
        tex.append(r"There are " + str(ntypes) + r" block types (same function form within a type, different weights). "
                   r"A type map $c:\{1,\dots,L\}\to\{" + ",".join(r"\mathrm{" + b["code"] + "}" for b in tblocks) +
                   r"\}$ says which type layer $\ell$ is: $\mathcal{B}^{(\ell)}=\mathcal{B}_{c(\ell)}(\,\cdot\,;\theta_\ell)$. "
                   r"Here $c=(" + pattern_str(text_stack["groups"]) + r")$, read left to right from layer 1.")
    tex.append(r"Written out, with the type as a subscript:")
    tex.append(r"\begin{equation}")
    tex.append(r"  f_\theta = \Psi\circ " + r"\circ ".join(seq) + r"\circ " + inp)
    tex.append(r"\end{equation}")
    tex.append(r"Grouping the repetitions with composition powers ($\mathcal{B}^{\circ n}$ = $n$ consecutive blocks of the "
               r"same type, each with its own weights) gives the compact form")
    tex.append(r"\begin{equation}\label{eq:compact}")
    tex.append(r"  f_\theta = \Psi\circ\Big(" + stack_expr(text_stack["groups"]) + r"\Big)\circ " + inp)
    tex.append(r"\end{equation}")
    tex.append(r"\end{definition}")

    for s in towers:
        enc = "Enc_V" if s["role"] == "vision" else "Enc_A"
        pre = "PatchEmbed" if s["role"] == "vision" else "Subsample"
        what = "image patches" if s["role"] == "vision" else "audio frames"
        dt = cget(s["cfg"], "hidden_size", "?")
        tex.append(r"\begin{definition}[" + ROLE_TITLE[s["role"]] + r"]")
        tex.append(r"The " + ROLE_TITLE[s["role"]].lower() + r" is itself a composition of $" + str(s["n"]) +
                   r"$ blocks acting on a sequence of " + what + r" of width $" + str(dt) + r"$:")
        tex.append(r"\begin{equation}")
        tex.append(r"  \op{" + enc + r"} = \op{Post}\circ " + stack_expr(s["groups"]) + r"\circ\op{" + pre + "}")
        tex.append(r"\end{equation}")
        tex.append(r"$\op{" + pre + r"}$ turns the raw input into vectors and $\op{Post}$ is the final norm / pooling / "
                   r"merging (Table~\ref{tab:nonrep}).")
        tex.append(r"\end{definition}")

    tex.append(r"\paragraph{Layer pattern.}")
    rows = []
    for s in stacks:
        for b in s["blocks"]:
            rows.append(f"${B(b['code'])}$ & {esc(b['name'])} & {len(b['layers'])} & {fmt(b['params'])} & "
                        f"{ranges_text(b['layers'])}")
    tex += table(r"l>{\raggedright\arraybackslash}p{5.6cm}rr>{\raggedright\arraybackslash}p{4.4cm}",
                 r"Symbol & Block type & Count & Params / block & Checkpoint indices", rows)
    for s in stacks:
        tex.append(r"\paragraph{" + ROLE_TITLE[s["role"]] + f" ({s['n']} layers).}} ")
        parts = []
        for unit, k, start in s["groups"]:
            u = ", ".join(f"${B(c)}$" + (f"$\\times{r}$" if r > 1 else "") for c, r in runs(unit))
            parts.append(f"layers {start}--{start + len(unit) * k - 1}: [{u}] repeated {k}$\\times$"
                         if k > 1 else f"layers {start}--{start + len(unit) - 1}: [{u}]")
        tex.append("; ".join(parts) + " (checkpoint indices, i.e.\\ $\\ell-1$).")

    # 2.2 the layers --------------------------------------------------------------
    tex.append(r"\subsection{The layers, top to bottom}\label{sec:layers}")
    tex.append(r"Every function in the composition above is defined here. Only the attention (or linear-attention) "
               r"sub-layer mixes information \emph{between} positions $t$. Every other sub-layer (norms, feed-forward "
               r"networks, projections) acts on each token vector $h_t\in\mathbb{R}^{d}$ independently, e.g.\ "
               r"$\op{FFN}(h)_t=\op{FFN}(h_t)$. All weights follow the convention $W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$, "
               r"so a projection of a token vector is $W u$ (as in an affine layer $f_i(u)=A^{(i)}u+b^{(i)}$).")

    tex.append(r"\begin{definition}[Token embedding]\label{def:embed}")
    tex.append(r"$E:\{1,\dots,V\}^{T}\to\mathbb{R}^{T\times d}$ looks up one row of the embedding matrix "
               r"$W_E\in\mathbb{R}^{V\times d}$ per token:")
    tex.append(r"\begin{equation}")
    tex.append(r"  E(x)_t = W_E[x_t,:]" + (r"\cdot\sqrt{d}" if gemma_scale else "") + r",\qquad t=1,\dots,T.")
    tex.append(r"\end{equation}")
    if towers:
        merge = [r"E(x)"]
        for s in towers:
            tag = "img" if s["role"] == "vision" else "aud"
            enc = "Enc_V" if s["role"] == "vision" else "Enc_A"
            merge.append(r"\op{Proj}_{\mathrm{" + tag + r"}}\big(\op{" + enc + r"}(\mathrm{" + tag + r"})\big)")
        tex.append(r"With images" + (r" or audio" if any(s["role"] == "audio" for s in towers) else "") +
                   r", the multimodal embedding replaces the placeholder-token rows of $E(x)$ by projected encoder outputs:")
        tex.append(r"\begin{equation}")
        tex.append(r"  E_{\mathrm{mm}}(x,\cdot) = \op{Merge}\Big(" + r",\ ".join(merge) + r"\Big).")
        tex.append(r"\end{equation}")
    if has_ple:
        tex.append(r"Per-layer embeddings (PLE): a second table gives every token one small vector for each layer,")
        tex.append(r"\begin{equation}")
        tex.append(r"  p = \tfrac{1}{\sqrt2}\Big(\op{Embed}_{\mathrm{PLE}}(x) + \op{RMSNorm}\big(W_{\mathrm{ple}}\,E(x)/\sqrt{d}\big)\Big),"
                   r"\qquad p=(p_1,\dots,p_L),\ p_\ell\in\mathbb{R}^{T\times " + str(f0["ple_dim"]) + r"}.")
        tex.append(r"\end{equation}")
    tex.append(r"\end{definition}")

    N_def = (r"$\op{RMSNorm}(u)=\gamma\odot u/\sqrt{\tfrac1d\sum_i u_i^2+\epsilon}$, with a learned scale $\gamma\in\mathbb{R}^{d}$")
    tex.append(r"\begin{definition}[Normalisation]")
    tex.append(N_def + r", applied to each token vector separately. It fixes the scale of a vector without changing its "
               r"direction. Every norm in the model has its own $\gamma$, distinguished by a subscript "
               r"(e.g.\ $\op{RMSNorm}_1$, $\op{RMSNorm}_{\mathrm{final}}$)." +
               (r" $\op{LN}$ (LayerNorm, used in some encoder blocks) additionally subtracts the mean and adds a bias." 
                if any(b["f"]["ln"] for s in stacks for b in s["blocks"]) else ""))
    tex.append(r"\end{definition}")

    for s in stacks:
        for b in s["blocks"]:
            f = b["f"]
            comp, g, scale = block_composition(f)
            tex.append(r"\begin{definition}[Block $" + B(b["code"]) + r"$: " + esc(b["name"]) + r"]")
            where = {"text": "decoder", "vision": "vision-encoder", "audio": "audio-encoder"}.get(s["role"], "")
            tex.append(r"A " + where + r" block of this type maps $h\in\mathbb{R}^{T\times " + str(f["d"]) + r"}$ to the same "
                       r"space. It is a composition of " + str(len(g)) + r" residual sub-layers, each of the form "
                       r"$(\mathrm{id}+g)(h)=h+g(h)$" + (r", followed by a final norm" if f["conformer"] else "") +
                       (r", scaled by the fixed per-layer scalar $\gamma_\ell$" if scale else "") + ":")
            tex.append(r"\begin{equation}")
            tex.append(r"  " + B(b["code"]) + r" = " + comp)
            tex.append(r"\end{equation}")
            tex.append(r"Step by step, the sub-layers are:")
            lines = wiring_lines(f)
            if not f["conformer"]:
                lines += mixer_lines(f) + ffn_lines(f)
            tex += align(lines)
            ps, rest = block_params(b)
            tex.append(r"Parameters of one block ($|\theta_\ell|=" + mnum(b["params"]) + r"$): " +
                       ", ".join(f"${p}$" for p in ps) +
                       (f", plus {fmt(rest)} parameters in norm scales, biases and other vectors" if rest > 0 else "") +
                       r". This type is used " + (r"once" if len(b["layers"]) == 1 else f"{len(b['layers'])} times") + ".")
            tex.append(r"\end{definition}")

    head = (r"W_E^{\top}" if tied else r"W_{\mathrm{LM}}") + r"\,\op{RMSNorm}_{\mathrm{final}}(h)_t"
    tex.append(r"\begin{definition}[Output head]")
    tex.append(r"$\Psi:\mathbb{R}^{T\times d}\to\mathbb{R}^{T\times V}$ normalises the last hidden state and scores every "
               r"vocabulary token" + (r", re-using the embedding matrix (tied weights)" if tied else
                                      r" with its own matrix $W_{\mathrm{LM}}\in\mathbb{R}^{V\times d}$") + ":")
    tex.append(r"\begin{equation}")
    if softcap:
        tex.append(r"  \Psi(h)_t = " + str(softcap) + r"\cdot\tanh\!\big(" + head + "/" + str(softcap) + r"\big)")
    else:
        tex.append(r"  \Psi(h)_t = " + head)
    tex.append(r"\end{equation}")
    if softcap:
        tex.append(r"The $\tanh$ soft-cap keeps every logit in $(-" + str(softcap) + "," + str(softcap) + ")$.")
    tex.append(r"\end{definition}")

    # 2.3 neurons and heads ------------------------------------------------------
    tex.append(r"\subsection{Neurons and attention heads}")
    act = f0["act"]
    ffn_blocks = [b for b in tblocks if b["f"]["ffn"]]
    if ffn_blocks and ffn_blocks[0]["f"]["ffn"] == "moe":
        fb = ffn_blocks[0]["f"]
        Ntot = sum(len(b["layers"]) * b["f"]["E"] * b["f"]["dff"] for b in ffn_blocks)
        Nact = sum(len(b["layers"]) * b["f"]["topk"] * b["f"]["dff"] for b in ffn_blocks)
        tex.append(r"\begin{definition}[Neuron, pre-activation value and state]")
        tex.append(r"Each block has $E=" + str(fb["E"]) + r"$ experts, each with $d_{ff}=" + str(fb["dff"]) + r"$ neurons. "
                   r"Neuron $j$ of expert $e$ in layer $\ell$ is the scalar function of a normalised token vector $u\in\mathbb{R}^{d}$")
        tex.append(r"\begin{equation}")
        tex.append(r"  \eta^{(\ell)}_{e,j}(u) = (c_j+1)\,a_j\,\sigma(1.702\,a_j),\qquad a_j = W^{g,(\ell)}_{e,j}\cdot u + b^{g}_{e,j},\quad c_j = W^{u,(\ell)}_{e,j}\cdot u + b^{u}_{e,j},")
        tex.append(r"\end{equation}")
        tex.append(r"where $W^{g}_{e,j}$, $W^{u}_{e,j}$ are the gate and up rows of $W^{gu}_e$. An expert's output is "
                   r"$\op{E}_e(u)=\sum_j \eta_{e,j}(u)\,W^{d}_{e}[:,j]+b^{d}_e$. The network has $N=" + mnum(Ntot) +
                   r"$ expert neurons, of which only $" + mnum(Nact) + r"$ (the top-" + str(fb["topk"]) +
                   r" experts in every layer) are evaluated for a given token. For input $x$ at position $t$:")
        pre = r"a_j"
    elif ffn_blocks:
        Ntot = sum(len(b["layers"]) * b["f"]["dff"] for b in ffn_blocks)
        dffs = valset(b["f"]["dff"] for b in ffn_blocks)
        tex.append(r"\begin{definition}[Neuron, pre-activation value and state]")
        tex.append(r"The feed-forward sub-layer of block $\ell$ contains $d_{ff}^{(\ell)}" + ("=" if "{" not in dffs else r"\in") + dffs +
                   r"$ neurons. Neuron $j$ of layer $\ell$ is the scalar function of a normalised token vector $u\in\mathbb{R}^{d}$")
        tex.append(r"\begin{equation}")
        tex.append(r"  \eta^{(\ell)}_j(u) = " + act + r"\big(W^{(\ell)}_{\mathrm{gate},j}\cdot u\big)\;\big(W^{(\ell)}_{\mathrm{up},j}\cdot u\big),")
        tex.append(r"\end{equation}")
        tex.append(r"where $W_{\mathrm{gate},j}$ is the $j$-th row of $W_{\mathrm{gate}}$. The feed-forward output is the "
                   r"sum of the neuron outputs, each written along its own column of $W_{\mathrm{down}}$:")
        tex.append(r"\begin{equation}")
        tex.append(r"  \op{FFN}_\ell(u) = \sum_{j=1}^{d_{ff}^{(\ell)}} \eta^{(\ell)}_j(u)\;W^{(\ell)}_{\mathrm{down}}[:,j].")
        tex.append(r"\end{equation}")
        tex.append(r"Across its $L$ blocks the network has $N=\sum_{\ell=1}^{L} d_{ff}^{(\ell)}=" + mnum(Ntot) +
                   r"$ feed-forward neurons. For input $x$ at position $t$, with $u^{(\ell)}_t(x)$ the normalised input of the feed-forward sub-layer:")
        pre = r"W^{(\ell)}_{\mathrm{gate},j}\cdot u^{(\ell)}_t(x)"
    if ffn_blocks:
        tex.append(r"\begin{itemize}")
        tex.append(r"\item $\mathcal{V}(\eta;x,t)=" + pre + r"\in\mathbb{R}$ is the \textbf{pre-activation value} of neuron $\eta$ (the argument of the non-linearity).")
        tex.append(r"\item $\mathcal{S}(\eta;x,t)$ is its \textbf{activation state}: \emph{active} if $\mathcal{V}>0$ and "
                   r"\emph{inactive} if $\mathcal{V}\le 0$. The non-linearity is smooth, unlike ReLU, so an inactive neuron's "
                   r"output is strongly suppressed rather than exactly zero.")
        tex.append(r"\end{itemize}")
        tex.append(r"\end{definition}")

    attn_blocks = [b for b in tblocks if b["f"]["mixer"] == "attn"]
    if attn_blocks:
        nat = sum(len(b["layers"]) for b in attn_blocks)
        fa = attn_blocks[0]["f"]
        gated = any(b["f"].get("gated") for b in attn_blocks)
        tex.append(r"\begin{definition}[Attention head and attention pattern]")
        tex.append(r"Head $h$ of an attention layer $\ell$ reads from the whole sequence through its attention pattern "
                   r"$P^{(\ell)}_h(x)\in\mathbb{R}^{T\times T}$ (row $t$ = how much position $t$ attends to each earlier position) "
                   r"and writes through its own $d\times d_h$ slice $W^{(\ell)}_{O,h}$ of $W_O$:")
        tex.append(r"\begin{equation}")
        tex.append(r"  \op{Attn}_\ell(u) = \sum_{h=1}^{H} " + (r"\big(P^{(\ell)}_h V_{g(h)}\odot\sigma(G_h)\big)" if gated else r"P^{(\ell)}_h\,V_{g(h)}") +
                   r"\,W^{(\ell)\top}_{O,h}" + (r" + b_O" if fa["o_bias"] else "") + ".")
        tex.append(r"\end{equation}")
        tex.append(r"$g(h)$ is the key/value head shared by query head $h$ (grouped-query attention). The pattern $P_h$ plays the "
                   r"role that the activation state plays for neurons: once $P_h$ is fixed, the head is linear in its values. "
                   r"The network has " + str(nat) + r" attention layers $\times$ $H=" + str(fa["H"]) + r"$ heads $=" +
                   str(nat * fa["H"]) + r"$ attention heads.")
        tex.append(r"\end{definition}")

    # 2.4 architecture and parameters ---------------------------------------------
    tex.append(r"\subsection{Architecture and parameters}")
    arch = [("V", Vs), ("d", str(d)), ("L", str(L)), ("c", "(" + pattern_str(text_stack["groups"]) + ")")]
    if attn_blocks:
        arch += [("H", valset(b["f"]["H"] for b in attn_blocks)), ("H_{kv}", valset(b["f"]["Hkv"] for b in attn_blocks)),
                 ("d_h", valset(b["f"]["dh"] for b in attn_blocks))]
    if ffn_blocks:
        arch.append(("d_{ff}", valset(b["f"]["dff"] for b in ffn_blocks)))
        if ffn_blocks[0]["f"]["ffn"] == "moe":
            arch += [("E", str(ffn_blocks[0]["f"]["E"])), ("k", str(ffn_blocks[0]["f"]["topk"]))]
    tex.append(r"\begin{definition}[Architecture and parameters]")
    tex.append(r"The \textbf{architecture} is the tuple of sizes and the layer-type sequence:")
    tex.append(r"\begin{equation}")
    tex.append(r"  \mathcal{A} = (" + ",".join(a for a, _ in arch) + r") = (" + ",\\ ".join(v for _, v in arch) + ").")
    tex.append(r"\end{equation}")
    rest_names = [r"W_E"] + [r"\theta_1,\dots,\theta_L", r"\gamma_{\mathrm{final}}"] + ([] if tied else [r"W_{\mathrm{LM}}"])
    rest_names += [r"\theta_{\op{" + ("Enc_V" if s["role"] == "vision" else "Enc_A") + "}}" for s in towers]
    if towers:
        rest_names.append(r"\theta_{\op{Proj}}")
    if has_ple:
        rest_names.append(r"\theta_{\mathrm{PLE}}")
    tex.append(r"The \textbf{parameters} are $\theta=\{" + ",\\ ".join(rest_names) + r"\}$" +
               (r" ($W_E$ is shared with the output head)" if tied else "") + r". Grouping the blocks by type,")
    stack_total = sum(b["params"] * len(b["layers"]) for s in stacks for b in s["blocks"])
    terms = [f"{len(b['layers'])}\\times{mnum(b['params'])}" for s in stacks for b in s["blocks"]]
    terms.append(mnum(total - stack_total))
    tex.append(r"\begin{align*}")
    tex.append(r"  |\theta| &= \sum_{\text{types } c} n_c\,|\theta_c| + |\theta_{\mathrm{rest}}| \\")
    for i in range(0, len(terms), 3):
        tex.append(r"  &" + ("=" if i == 0 else r"\quad+") + " " + " + ".join(terms[i:i + 3]) + r" \\")
    tex.append(r"  &= " + mnum(total) + ",")
    tex.append(r"\end{align*}")
    tex.append(r"where $n_c$ is how often type $c$ occurs and $\theta_{\mathrm{rest}}$ holds everything outside the repeated "
               r"stacks (embedding, final norm, " + ("encoder stems and projector, " if towers else "") +
               r"\dots; Table~\ref{tab:nonrep}).")
    tex.append(r"\end{definition}")

    # 2.5 outputs --------------------------------------------------------------------
    tex.append(r"\subsection{Output representations}")
    tex.append(r"\begin{definition}[Output representations]")
    tex.append(r"For an input sequence $x$, the output can be read at three levels:")
    tex.append(r"\begin{enumerate}")
    tex.append(r"\item \textbf{Logits:} $y=f_\theta(x)\in\mathbb{R}^{T\times V}$, one score per vocabulary token at every position.")
    tex.append(r"\item \textbf{Next-token distribution:} $p(x)_{t,k}=\dfrac{\exp(y_{t,k})}{\sum_{j=1}^{V}\exp(y_{t,j})}"
               r"=\Pr\big(x_{t+1}=k\mid x_{1},\dots,x_{t}\big)$.")
    tex.append(r"\item \textbf{Predicted next token:} $z(x)=\operatorname{argmax}_{k\in\{1,\dots,V\}} y_{T,k}$ (greedy decoding). "
               r"Generation appends $z(x)$ (or a sample from $p(x)_{T}$) to $x$ and applies $f_\theta$ again; in practice "
               r"the keys and values of earlier positions are cached.")
    tex.append(r"\end{enumerate}")
    tex.append(r"The model is \textbf{causal}: $y_t$ depends only on $x_1,\dots,x_t$, because every attention mask hides "
               r"later positions" + (r" (sliding-window layers additionally hide positions more than $w$ tokens back)"
                                     if any(b["f"]["sliding"] for b in tblocks) else "") +
               r". Training minimises the cross-entropy $-\sum_{t}\log p(x)_{t,x_{t+1}}$.")
    tex.append(r"\end{definition}")

    # 2.6 intermediate states -------------------------------------------------------
    tex.append(r"\subsection{Intermediate states and sub-network decomposition}")
    tex.append(r"To study what happens at one layer $\ell\in\{1,\dots,L\}$, split the composition into a prefix, the "
               r"layer itself and a suffix:")
    tex.append(r"\begin{equation}")
    tex.append(r"  f_\theta = \underbrace{\Psi\circ\mathcal{B}^{(L)}\circ\cdots\circ\mathcal{B}^{(\ell+1)}}_{G_{\ell+1}\ \text{(suffix)}}"
               r"\circ\underbrace{\mathcal{B}^{(\ell)}}_{\text{layer }\ell}\circ"
               r"\underbrace{\mathcal{B}^{(\ell-1)}\circ\cdots\circ\mathcal{B}^{(1)}\circ " + inp + r"}_{F_{\ell-1}\ \text{(prefix)}}")
    tex.append(r"\end{equation}")
    M = r"\op{Mixer}" if f0["mixer"] == "gdn" else r"\op{Attn}"
    Nn = r"\op{RMSNorm}"
    if f0["sandwich"]:
        a_def = Nn + r"_{\mathrm{pa}}\big(" + M + r"_\ell(" + Nn + r"_{\mathrm{in}}(h^{(\ell-1)}))\big)"
        u_def = Nn + r"_{\mathrm{pre}}(\hat h^{(\ell)})"
        m_def = Nn + r"_{\mathrm{pf}}\big(\op{FFN}_\ell(u^{(\ell)})\big)"
    else:
        a_def = M + r"_\ell\big(" + Nn + r"_{1}(h^{(\ell-1)})\big)"
        u_def = Nn + r"_{2}(\hat h^{(\ell)})"
        m_def = r"\op{FFN}_\ell(u^{(\ell)})"
    tex.append(r"\begin{itemize}")
    tex.append(r"\item \textbf{Prefix} $F_{\ell-1}:\mathcal{X}\to\mathbb{R}^{T\times d}$ maps the tokens to the hidden state "
               r"entering layer $\ell$ ($F_0=" + inp + r"$).")
    tex.append(r"\item \textbf{Hidden state} (residual stream) after layer $\ell$: $h^{(\ell)}=F_\ell(x)=\mathcal{B}^{(\ell)}(h^{(\ell-1)}" + ple_arg + r")\in\mathbb{R}^{T\times d}$, with $h^{(0)}=" + inp + r"(x)$.")
    tex.append(r"\item \textbf{Inside layer $\ell$:} the attention write $a^{(\ell)}=" + a_def + r"$, the mid-block state "
               r"$\hat h^{(\ell)}=h^{(\ell-1)}+a^{(\ell)}$, the feed-forward input $u^{(\ell)}=" + u_def + r"$, the "
               r"pre-activations $\hat x^{(\ell)}=u^{(\ell)}W^{(\ell)\top}_{\mathrm{gate}}\in\mathbb{R}^{T\times d_{ff}}$ and the "
               r"neuron outputs $\eta^{(\ell)}(u^{(\ell)})\in\mathbb{R}^{T\times d_{ff}}$, and the feed-forward write "
               r"$m^{(\ell)}=" + m_def + r"$." if f0["ffn"] == "gated" else
               r"\item \textbf{Inside layer $\ell$:} the attention write $a^{(\ell)}=" + a_def + r"$, the mid-block state "
               r"$\hat h^{(\ell)}=h^{(\ell-1)}+a^{(\ell)}$, the feed-forward input $u^{(\ell)}=" + u_def + r"$ and the "
               r"feed-forward write $m^{(\ell)}=" + m_def + r"$.")
    tex.append(r"\item \textbf{Suffix} $G_{\ell+1}:\mathbb{R}^{T\times d}\to\mathbb{R}^{T\times V}$ maps $h^{(\ell)}$ to the "
               r"logits ($G_{L+1}=\Psi$).")
    tex.append(r"\end{itemize}")
    tex.append(r"\paragraph{Residual stream.} Because every sub-layer only \emph{adds} to the hidden state, unrolling the "
               r"composition gives the final hidden state as the embedding plus everything the layers wrote:")
    tex.append(r"\begin{equation}")
    extra = r"+c^{(\ell)}" if has_ple else ""
    if f0["layer_scalar"]:
        tex.append(r"  h^{(L)} = \Big(\prod_{k=1}^{L}\gamma_k\Big)\,h^{(0)} + \sum_{\ell=1}^{L}\Big(\prod_{k=\ell}^{L}\gamma_k\Big)"
                   r"\big(a^{(\ell)}+m^{(\ell)}" + extra + r"\big)")
    else:
        tex.append(r"  h^{(L)} = h^{(0)} + \sum_{\ell=1}^{L}\big(a^{(\ell)}+m^{(\ell)}" + extra + r"\big)")
    tex.append(r"\end{equation}")
    if has_ple:
        tex.append(r"with $c^{(\ell)}=\op{PLE}_\ell(\hat h^{(\ell)}+m^{(\ell)})$ the per-layer-embedding write.")
    if f0["layer_scalar"]:
        tex.append(r"The fixed scalars $\gamma_\ell$ rescale everything written so far, so earlier writes are multiplied by all later scalars.")
    tex.append(r"So $f_\theta(x)=\Psi(h^{(L)})$: each layer reads the current stream and adds a correction to it.")

    # 2.7 local linear --------------------------------------------------------------
    tex.append(r"\subsection{Local linear representation}")
    tex.append(r"A ReLU network is piecewise affine: inside a polytope cell it collapses to one matrix. This model is not, because "
               r"softmax, $" + act + r"$ and the normalisations are smooth. Its local behaviour is described by the first-order "
               r"(Jacobian) linearisation instead. Tokens are discrete, so we perturb the embedded input $h^{(0)}$.")
    tex.append(r"\begin{definition}[Local linear response]")
    tex.append(r"Let $J_\ell=D\mathcal{B}^{(\ell)}(h^{(\ell-1)})$ be the Jacobian of block $\ell$ at the current hidden state "
               r"(tensors flattened to vectors). The chain rule applied to the residual sub-layers $(\mathrm{id}+g_{\ell,i})$ gives")
    nb = len(block_composition(f0)[1])
    prod = r"\cdots".join(f"(I+Dg_{{\\ell,{i}}})" for i in range(nb, 0, -1)) if nb <= 3 else r"\prod_{i}(I+Dg_{\ell,i})"
    tex.append(r"\begin{equation}")
    tex.append(r"  J_\ell = " + (r"\gamma_\ell\," if f0["layer_scalar"] else "") + prod.replace(r")\cdots(", r")(") + r",")
    tex.append(r"\end{equation}")
    tex.append(r"where $g_{\ell,1}$ is the attention branch, $g_{\ell,2}$ the feed-forward branch" +
               (r", $g_{\ell,3}$ the PLE branch" if has_ple else "") + r" and each $Dg$ is taken at that branch's input. "
               r"For a small perturbation $\delta$ of $h^{(0)}$, the response of the network decomposed around layer $\ell$ is")
    tex.append(r"\begin{equation}")
    tex.append(r"  f_\theta(h^{(0)}+\delta)-f_\theta(h^{(0)}) \approx G^{(\ell+1)}_x\cdot J_\ell\cdot F^{(\ell-1)}_x\cdot\delta,")
    tex.append(r"\end{equation}")
    tex.append(r"with $F^{(\ell-1)}_x=J_{\ell-1}\cdots J_1$ (prefix) and $G^{(\ell+1)}_x=J_\Psi\,J_L\cdots J_{\ell+1}$ (suffix). "
               r"Multiplying out the products of $(I+Dg)$ terms writes $J_\Psi J_L\cdots J_1$ as a sum over \emph{paths}: the "
               r"identity path straight from embedding to head, plus one term for every subset of sub-layers a signal passes through.")
    tex.append(r"\end{definition}")

    # 2.8 notation -------------------------------------------------------------------
    tex.append(r"\subsection{Summary of notation}")
    rows = [r"$f_\theta:\mathcal{X}\to\mathcal{Y}$ & the whole model, $\{1..V\}^T\to\mathbb{R}^{T\times V}$",
            r"$" + inp + r",\ \Psi$ & embedding and output head",
            r"$\mathcal{B}^{(\ell)},\ \theta_\ell$ & block $\ell$ and its parameters, $\ell=1,\dots,L$",
            r"$\mathcal{B}_c,\ c(\ell)$ & block type $c$; type of layer $\ell$",
            r"$\mathcal{B}^{\circ n}$ & $n$ consecutive blocks of one type (each with its own weights)",
            r"$(\mathrm{id}+g)$ & residual sub-layer $h\mapsto h+g(h)$",
            r"$h^{(\ell)}$ & hidden state (residual stream) after layer $\ell$; $h^{(0)}$ = embeddings",
            r"$a^{(\ell)},\ m^{(\ell)}$ & what the attention / feed-forward sub-layer of layer $\ell$ adds to the stream",
            r"$\eta^{(\ell)}_j$ & $j$-th feed-forward neuron of layer $\ell$",
            r"$\mathcal{V}(\eta;x,t),\ \mathcal{S}(\eta;x,t)$ & pre-activation value and active/inactive state of a neuron",
            r"$P^{(\ell)}_h$ & attention pattern of head $h$ in layer $\ell$ ($T\times T$)",
            r"$F_{\ell-1},\ G_{\ell+1}$ & prefix (embedding to layer $\ell-1$) and suffix (layer $\ell+1$ to logits)",
            r"$J_\ell,\ F^{(\ell-1)}_x,\ G^{(\ell+1)}_x$ & Jacobian of block $\ell$, and of the prefix / suffix at input $x$",
            r"$p(x),\ z(x)$ & next-token distribution and greedy next-token prediction"]
    if has_ple:
        rows.insert(3, r"$p_\ell$ & per-layer embedding fed to block $\ell$")
    if towers:
        rows.insert(2, r"$\op{Enc_V},\ \op{Enc_A},\ \op{Proj}$ & vision / audio encoders and their projectors into the text width")
    tex += table(r"l>{\raggedright\arraybackslash}p{10.5cm}", r"Symbol & Meaning", rows)
    return tex


def layer_loops(groups, names, ple=False):
    """Pseudocode loops over a stack; layers are numbered l = 1..n as in the math section."""
    out = []
    arg = r"h,\,p_\ell" if ple else "h"
    call = lambda c: r"h \gets " + B(c) + "(" + arg + r";\,\theta_\ell)"
    for unit, k, start in groups:
        rs = runs(unit)
        lo, hi = start + 1, start + len(unit) * k
        if len(rs) == 1:
            c, n_ = rs[0]
            if n_ * k > 1:
                out.append(r"\For{$\ell \gets " + str(lo) + r"$ \textbf{to} $" + str(hi) + r"$} \Comment{" +
                           f"{n_ * k} blocks of type ${B(c)}$" + "}")
                out.append(r"\State $" + call(c) + r"$ \Comment{" + names[c] + r"; result is $h^{(\ell)}$}")
                out.append(r"\EndFor")
            else:
                out.append(r"\State $\ell \gets " + str(lo) + r";\ " + call(c) + r"$ \Comment{" + names[c] + "}")
            continue
        out.append(r"\State $\ell \gets " + str(lo - 1) + "$")
        if k > 1:
            out.append(r"\For{$r \gets 1$ \textbf{to} $" + str(k) + r"$} \Comment{layers " +
                       f"{lo}--{hi}: pattern of {len(unit)} blocks, repeated {k} times" + "}")
        for c, n_ in rs:
            step = r"\State $\ell \gets \ell+1;\ " + call(c) + r"$ \Comment{" + names[c] + "}"
            if n_ > 1:
                out.append(r"\For{$j \gets 1$ \textbf{to} $" + str(n_) + r"$}")
                out += [step, r"\EndFor"]
            else:
                out.append(step)
        if k > 1:
            out.append(r"\EndFor")
    return out


def generate_tex(model_name, model_path, model, config, cls, recs, types, variants=()):
    total = sum(r["numel"] for r in recs if not r["tied_to"])
    text_cfg = config.get_text_config()
    stacks = [analyse_stack(model, config, s, recs, types) for s in find_stacks(model, types)]
    stacks.sort(key=lambda s: s["role"] != "text")
    text_stack = next((s for s in stacks if s["role"] == "text"), None)
    towers = [s for s in stacks if s["role"] != "text"]
    in_stack = lambda p: any(p.startswith(s["path"] + ".") for s in stacks)
    for s in stacks:
        for b in s["blocks"]:
            b["f"] = block_features(model, s["path"], b, s["role"], s["cfg"], types)
            b["params"] = sum(r["numel"] for r in b["recs"])
    sandwich = any(b["f"]["sandwich"] for b in (text_stack["blocks"] if text_stack else []))
    tied = any(r["tied_to"] for r in recs if r["path"].startswith("lm_head"))
    ckpt_extra = checkpoint_only_tensors(model_path, types)
    name_tex = esc(model_name)
    d = cget(text_cfg, "hidden_size", "d")
    V = cget(text_cfg, "vocab_size", "V")
    has_ple = any(b["f"]["ple"] for b in (text_stack["blocks"] if text_stack else []))
    softcap = cget(text_cfg, "final_logit_softcapping", None)
    gemma_scale = "Gemma" in cls

    tex = [PREAMBLE]
    tex.append(r"\title{Model Architecture: " + name_tex + "}")
    tex.append(r"\author{Generated by sg\_modelload}")
    tex.append(r"\date{\today}")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")
    tex.append(r"\tableofcontents")
    tex.append(r"\clearpage")

    # 1. Overview -------------------------------------------------------------
    tex.append(r"\section{Model Overview}")
    tex.append(r"\begin{tabular}{ll}")
    tex.append(r"\toprule")
    tex.append(f"Model name & {name_tex} \\\\")
    tex.append(f"Transformers class & {tt(cls)} \\\\")
    tex.append(f"Path & {tt(model_path)} \\\\")
    for vname, vpath in variants:
        tex.append(f"Same architecture as & {esc(vname)} \\\\")
        tex.append(f"\\quad its path & {tt(vpath)} \\\\")
    tex.append(f"Total parameters & {fmt(total)} ({human(total)}) \\\\")
    if ckpt_extra:
        tex.append(f"Checkpoint-only tensors (not in total) & {fmt(sum(r['numel'] for r in ckpt_extra))} \\\\")
    tex.append(f"Weight dtype & {esc(recs[0]['dtype'])} \\\\")
    q = cget(text_cfg, "quantization_config", None) or cget(config, "quantization_config", None)
    if q:
        qm = q.get("quant_method") if isinstance(q, dict) else getattr(q, "quant_method", "")
        tex.append(f"Checkpoint quantisation & {esc(qm)} (shapes below are the de-quantised ones) \\\\")
    tex.append(r"\midrule")
    for r in hyper_rows(config):
        tex.append(r + r" \\")
    for s in towers:
        c = s["cfg"]
        tex.append(r"\midrule")
        tex.append(f"{ROLE_TITLE[s['role']]}: layers & {s['n']} \\\\")
        tex.append(f"\\quad hidden size / heads & {cget(c, 'hidden_size', '?')} / "
                   f"{cget(c, 'num_attention_heads', None) or cget(c, 'num_heads', '?')} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    if variants:
        names = ", ".join(esc(v) for v, _ in variants)
        tex.append(r"\paragraph{Same structure.} " + names + r" has exactly the same architecture as " + name_tex +
                   r" (same class, every tensor name and shape identical, checked when this document was generated). "
                   r"Fine-tuning changes only the weight values, so everything in this document applies to it unchanged. "
                   r"The two configs differ only in generation settings such as \texttt{eos\_token\_id}.")
    tex.append(r"\paragraph{How to read this document.} Section~\ref{sec:func} treats the model mathematically, top-down: "
               r"first the whole network as one composition $\Psi\circ\mathcal{B}^{(L)}\circ\cdots\circ\mathcal{B}^{(1)}\circ E$ "
               r"of layers, then a definition of every layer down to single neurons and attention heads, then parameters, "
               r"outputs, intermediate states and the local linear response. Section~\ref{sec:algo} gives readable pseudocode for the forward pass and for every "
               r"distinct block type. Section~\ref{sec:compact} lists the weights of each block type \emph{once}, with "
               r"how many times it repeats. Section~\ref{sec:gloss} explains in plain language what every kind of layer "
               r"does. The original layer-by-layer table with every repetition written out is in "
               r"Appendix~\ref{app:full}.")

    # Notation ----------------------------------------------------------------
    tex.append(r"\subsection*{Basic notation}")
    tex.append(r"\begin{tabular}{ll}")
    tex.append(r"\toprule")
    tex.append(r"$B$, $T$ & batch size, sequence length (tokens) \\")
    tex.append(f"$d$ & hidden size ({d}) \\\\")
    tex.append(f"$V$ & vocabulary size ({fmt(V) if isinstance(V, int) else V}) \\\\")
    tex.append(r"$H,\ H_{kv},\ d_h$ & query heads, key/value heads, size of one head \\")
    tex.append(r"$W_\ast,\ b_\ast$ & learned weight matrix / bias of the named projection \\")
    tex.append(r"$\odot$, $\sigma$ & element-wise product, logistic sigmoid \\")
    tex.append(r"$f^{\circ n}$ & $f$ applied $n$ times in a row ($f\circ f\circ\cdots\circ f$) \\")
    tex.append(r"$g \circ f$ & first $f$, then $g$ (composition reads right to left) \\")
    tex.append(r"$\op{RMSNorm}(x)$ & $\gamma \odot x / \sqrt{\mathrm{mean}(x^2)+\epsilon}$ (learned scale $\gamma$) \\")
    tex.append(r"$\op{LN}(x)$ & LayerNorm: $\gamma\odot(x-\mu)/\sigma + \beta$ \\")
    tex.append(r"$\op{RoPE}$ & rotary position embedding (rotates query/key pairs by an angle set by position) \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    # 2. Mathematical representation -------------------------------------------------
    if text_stack:
        tex += math_section(name_tex, stacks, text_stack, towers, d, V, tied, softcap, has_ple, gemma_scale, total)

    # 3. Pseudocode ---------------------------------------------------------------
    tex.append(r"\section{Forward Pass in Pseudocode}\label{sec:algo}")
    if text_stack:
        tex.append(r"\begin{algorithm}[H]")
        tex.append(r"\caption{" + name_tex + r": full forward pass}")
        tex.append(r"\begin{algorithmic}[1]")
        tex.append(r"\Require token ids $x$ of shape $B\times T$" + (r"; optional images / audio" if towers else ""))
        tex.append(r"\State $h \gets E(x)" + (r"\cdot\sqrt{d}" if gemma_scale else "") +
                   f"$ \\Comment{{$h^{{(0)}}$: one vector per token, $B{{\\times}}T{{\\times}}{d}$}}")
        for s in towers:
            what = "image" if s["role"] == "vision" else "audio"
            enc = "Enc_V" if s["role"] == "vision" else "Enc_A"
            tex.append(r"\If{" + what + r" input present}")
            tex.append(r"\State $e \gets \op{" + enc + r"}(\text{" + what + r"})$ \Comment{" +
                       f"{ROLE_TITLE[s['role']].lower()}: {s['n']} layers, Algorithm~\\ref{{alg:{s['role']}}}" + "}")
            tex.append(r"\State $h[\text{" + what + r" token slots}] \gets \op{Proj}(e)$ \Comment{project into text width " + str(d) + "}")
            tex.append(r"\EndIf")
        if has_ple:
            tex.append(r"\State $p \gets \op{PerLayerInputs}(x, h)$ \Comment{" +
                       f"{text_stack['n']} small vectors per token, one for each layer" + "}")
        names = {b["code"]: esc(b["name"]) for b in text_stack["blocks"]}
        tex += layer_loops(text_stack["groups"], names, ple=has_ple)
        tex.append(r"\State $h \gets \op{RMSNorm}_{\mathrm{final}}(h)$ \Comment{$h=h^{(L)}$; this and the next step are $\Psi$}")
        tex.append(r"\State $y \gets h\," + (r"W_E^{\top}" if tied else r"W_{\mathrm{LM}}^{\top}") +
                   f"$ \\Comment{{logits $B{{\\times}}T{{\\times}}{fmt(V) if isinstance(V, int) else V}$" +
                   (", embedding matrix re-used" if tied else "") + "}")
        if softcap:
            tex.append(f"\\State $y \\gets {softcap}\\tanh(y/{softcap})$ \\Comment{{soft-cap logits}}")
        tex.append(r"\State \Return $y$ \Comment{softmax$(y)$ = next-token probabilities}")
        tex.append(r"\end{algorithmic}")
        tex.append(r"\end{algorithm}")

    for s in towers:
        tex.append(r"\begin{algorithm}[H]")
        tex.append(r"\caption{" + ROLE_TITLE[s["role"]] + r"}\label{alg:" + s["role"] + "}")
        tex.append(r"\begin{algorithmic}[1]")
        if s["role"] == "vision":
            tex.append(r"\State $h \gets \op{PatchEmbed}(\text{pixels}) + \text{position embedding}$ \Comment{one vector per image patch}")
        else:
            tex.append(r"\State $h \gets \op{Subsample}(\text{mel spectrogram})$ \Comment{2 strided convs: 4$\times$ fewer frames}")
        tex += layer_loops(s["groups"], {b["code"]: esc(b["name"]) for b in s["blocks"]})
        tex.append(r"\State $h \gets \op{Post}(h)$ \Comment{final norm / pooling / merging, see Table~\ref{tab:nonrep}}")
        tex.append(r"\State \Return $h$")
        tex.append(r"\end{algorithmic}")
        tex.append(r"\end{algorithm}")

    for s in stacks:
        for b in s["blocks"]:
            tex.append(r"\begin{algorithm}[H]")
            tex.append(r"\caption{Block $" + B(b["code"]) + r"(h)$ --- " + esc(b["name"]) + "}")
            tex.append(r"\begin{algorithmic}[1]")
            tex.append(r"\Require $h$ of shape $B\times T\times " + str(b["f"]["d"]) + "$" +
                       (r", per-layer input $p_\ell$" if b["f"]["ple"] else ""))
            tex += block_algorithm(b["f"], b["code"], b["name"], d)
            tex.append(r"\end{algorithmic}")
            tex.append(r"\end{algorithm}")

    # 4. Compact architecture -------------------------------------------------------
    tex.append(r"\section{Compact Architecture (repeated layers written once)}\label{sec:compact}")
    nonrep = [r for r in recs if not in_stack(r["path"])]
    tex.append(r"\subsection{Modules outside the repeated stacks}")
    rows = []
    for r in nonrep:
        note = f" (tied to {tt(r['tied_to'])})" if r["tied_to"] else ""
        rows.append(f"{tt(r['path'])}{note} & {ttype(r['type'])} & {shape_tex(r['shape'])} & "
                    f"{'0 (shared)' if r['tied_to'] else fmt(r['numel'])}")
    tex += table(r"p{6.9cm}>{\raggedright\arraybackslash}p{3.4cm}ll", r"Weight & Type & Shape & Params", rows,
                 caption=r"Weights that appear once\label{tab:nonrep}")
    for s in stacks:
        tex.append(r"\subsection{" + ROLE_TITLE[s["role"]] + r": \texttt{" + esc(s["path"]) + "}" + "}")
        for b in s["blocks"]:
            rows = [f"{tt(r['rel'])} & {ttype(r['type'])} & {shape_tex(r['shape'])} & {fmt(r['numel'])}" for r in b["recs"]]
            rows.append(r"\midrule \textbf{Per block} & & & \textbf{" + fmt(b["params"]) + "}")
            rows.append(r"\textbf{$\times$ " + str(len(b["layers"])) + r" blocks} & & & \textbf{" +
                        fmt(b["params"] * len(b["layers"])) + "}")
            tex += table(r"p{6.6cm}>{\raggedright\arraybackslash}p{3.6cm}ll", r"Weight (relative) & Type & Shape & Params",
                         rows, caption=f"${B(b['code'])}$: {esc(b['name'])} --- layers {ranges_text(b['layers'])}; "
                                       f"weight names are relative to \\texttt{{{esc(s['path'])}.$\\ell$}}")
    if ckpt_extra:
        tex.append(r"\subsection{Checkpoint-only modules}")
        for top in sorted({r["path"].split(".")[0] for r in ckpt_extra}):
            title, desc = CKPT_ONLY_DESC.get(top, (f"{top} (not instantiated by transformers)",
                                                   "Stored in the checkpoint but not built by the transformers class."))
            tex.append(r"\paragraph{" + esc(title) + r" (\texttt{" + esc(top) + ".*}).} " + desc)
            rows = [f"{tt(r['path'])} & {shape_tex(r['shape'])} & {esc(r['dtype'])} & {fmt(r['numel'])}"
                    for r in ckpt_extra if r["path"].split(".")[0] == top]
            tex += table(r"p{8.5cm}lll", r"Tensor & Shape & Dtype & Params", rows)

    # 5. Glossary -----------------------------------------------------------------------
    tex.append(r"\section{What Each Layer Does}\label{sec:gloss}")
    tex.append(r"One row per \emph{kind} of layer present in this model. ``Appears as'' gives the module names "
               r"(layer indices removed).")
    rows, group = [], None
    gated = any(b["f"].get("gated") for s in stacks for b in s["blocks"])
    for (g, title), (desc, names) in glossary_for(recs, sandwich, gated).items():
        if g != group:
            group = g
            rows.append((r"\midrule " if rows else "") + r"\multicolumn{2}{l}{\textsc{" + g + "}}")
        rows.append(r"\textbf{" + title + r"}\newline{\footnotesize " + ", ".join(tt(n) for n in names) + "} & " + desc)
    tex += table(r">{\raggedright\arraybackslash}p{5.4cm}>{\raggedright\arraybackslash}p{10.8cm}", r"Layer (appears as) & What it does", rows)

    # 6. Parameter counts ------------------------------------------------------------------
    tex.append(r"\section{Parameter Count}")
    comp = OrderedDict()
    for r in recs:
        if not r["tied_to"]:
            c = component_of(r["path"])
            comp[c] = comp.get(c, 0) + r["numel"]
    rows = [f"{c} & {fmt(n)} & {n / total * 100:.1f}\\%" for c, n in sorted(comp.items(), key=lambda x: -x[1])]
    rows.append(r"\midrule \textbf{Total} & \textbf{" + fmt(total) + r"} & \textbf{100.0\%}")
    tex += table("lrr", r"Component & Parameters & \% of total", rows, caption="Parameters by component")
    rows = []
    for s in stacks:
        for b in s["blocks"]:
            rows.append(f"${B(b['code'])}$ {esc(b['name'])} & {len(b['layers'])} & {fmt(b['params'])} & "
                        f"{fmt(b['params'] * len(b['layers']))}")
    stack_total = sum(b["params"] * len(b["layers"]) for s in stacks for b in s["blocks"])
    rows.append(f"Modules outside stacks & 1 & -- & {fmt(total - stack_total)}")
    rows.append(r"\midrule \textbf{Total} & & & \textbf{" + fmt(total) + "}")
    tex += table("p{8cm}rrr", r"Block type & Count & Params / block & Total", rows, caption="Parameters by block type")

    # Appendix: original tables --------------------------------------------------------------
    tex.append(r"\begin{appendices}")
    tex.append(r"\section{Full Layer-by-Layer Table}\label{app:full}")
    tex.append(r"Every weight tensor with every repetition written out (the original table).")
    rows = []
    seen_mod = set()
    for r in sorted(recs, key=lambda r: natural_key(r["path"])):
        mod = r["module"]
        parts = mod.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in seen_mod and types.get(parent) not in (None,):
                seen_mod.add(parent)
                if re.search(r"\.\d+$", parent) or types[parent] != "ModuleList":
                    rows.append(f"{tt(parent)} & {ttype(types[parent])} & -- & -- & --")
        tied_note = r" \textit{(tied)}" if r["tied_to"] else ""
        rows.append(f"{tt(r['path'])}{tied_note} & {ttype(r['type'])} & {shape_tex(r['shape'])} & {esc(r['dtype'])} & {fmt(r['numel'])}")
    tex.append(r"{\footnotesize")
    tex += table(r"p{6.8cm}>{\raggedright\arraybackslash}p{3.3cm}llr", r"Layer path & Type & Shape & Dtype & Params", rows)
    tex.append("}")
    tex.append(r"\section{Per-Layer Parameter Count}")
    rows, running = [], 0
    per_mod = OrderedDict()
    for r in sorted(recs, key=lambda r: natural_key(r["path"])):
        if not r["tied_to"]:
            per_mod[r["module"]] = per_mod.get(r["module"], 0) + r["numel"]
    for m, n in per_mod.items():
        running += n
        rows.append(f"{tt(m)} & {fmt(n)} & {fmt(running)}")
    tex += table(r"p{9.5cm}rr", r"Module & Params & Running total", rows)
    tex.append(r"\end{appendices}")
    tex.append(r"\end{document}")

    tex_path = os.path.join(OUT_DIR, f"{safe_name(model_name)}.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"  Written: {tex_path}")
    return tex_path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    only = set(sys.argv[1:])
    for model_name, model_path in models_info:
        if only and model_name not in only:
            continue
        print(f"\n{'=' * 60}\nAnalyzing: {model_name}\n{'=' * 60}")
        try:
            model, config, cls = build_model(model_path)
            recs, types = collect_records(model)
            sig = arch_signature(cls, recs)
            variants = []
            for vname, vpath in same_arch.get(model_name, []):
                vmodel, _, vcls = build_model(vpath)
                if arch_signature(vcls, collect_records(vmodel)[0]) == sig:
                    variants.append((vname, vpath))
                    print(f"  {vname}: identical architecture, covered by this report")
                else:
                    print(f"  WARNING: {vname} differs from {model_name}; give it its own entry in models_info")
            generate_tex(model_name, model_path, model, config, cls, recs, types, variants)
            del model
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
    print(f"\nAll .tex files saved to {OUT_DIR}/")
