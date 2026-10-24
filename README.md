# Model Architecture .tex Generation & Compilation

Generate detailed LaTeX architecture diagrams for transformer models, then compile to PDF.

## Required Files

- `sg_modelLoad` — CUDA + Python 3.12 + torch + transformers image
- `generate_tex.py` — loads each model and writes a `.tex` file with full layer-by-layer architecture
- `load_models.py` — utility to verify all models load correctly on GPU
- `Dockerfile.texlive` — TeX Live full image (compile `.tex` → `.pdf`)

## Order of Execution

Every container must carry `--label owner=$(whoami)` (cluster rule: unlabelled containers are killed).
Models are read from `/opt/dlami/nvme/sarbajit/local_models` (the `model` alias in `~/.bashrc`), mounted at `/models`.

### 1. Build the ML image
```bash
docker build --label owner=$(whoami) --build-arg OWNER=$(whoami) \
    -t sg_modelload -f sg_modelLoad .
```

### 2. Generate `.tex` files from all models
No GPU needed: each model is built on the meta device from its `config.json` (no weights loaded).
Pass model names (e.g. `python generate_tex.py gpt-oss-20b`) to generate only those.
```bash
docker run --rm --label owner=$(whoami) \
    --user "$(id -u):$(id -g)" \
    -v /opt/dlami/nvme/sarbajit/local_models:/models:ro \
    -v /opt/dlami/nvme/sarbajit/model_structure:/workspace \
    -w /workspace \
    sg_modelload \
    python generate_tex.py
```
Output: `tex_output/*.tex` (one per model)

### 3. TeX Live image
The existing `sg_tex` image already has `pdflatex`. To rebuild it:
```bash
docker build --label owner=$(whoami) --build-arg OWNER=$(whoami) \
    -t sg_tex -f Dockerfile.texlive .
```

### 4. Compile `.tex` → `.pdf`
Two passes so the table of contents and cross-references resolve.
```bash
docker run --rm --label owner=$(whoami) \
    --user "$(id -u):$(id -g)" \
    -v /opt/dlami/nvme/sarbajit/model_structure:/workdir \
    -w /workdir \
    sg_tex \
    sh -c 'mkdir -p tex_output/build tex_output/pdf && \
           for f in tex_output/*.tex; do \
             pdflatex -interaction=nonstopmode -output-directory=tex_output/build "$f" >/dev/null; \
             pdflatex -interaction=nonstopmode -output-directory=tex_output/build "$f" >/dev/null; \
           done && \
           mv tex_output/build/*.pdf tex_output/pdf/ && rm -rf tex_output/build'
```
Output: `tex_output/pdf/*.pdf`

## Output Structure

```
tex_output/
  ├── *.tex           (LaTeX source per model)
  └── pdf/
      └── *.pdf       (compiled PDFs)
```

## What Each `.tex` / `.pdf` Contains

1. **Model Overview**: parameter count, dtype, key hyperparameters (and vision/audio tower sizes).
2. **Mathematical Representation** (top-down, in the style of the formal DNN definition
   `f = f_{r+1} ∘ σ ∘ f_r ∘ … ∘ σ ∘ f_1`):
   - the model as a composition `f_θ = Ψ ∘ B^{(L)} ∘ … ∘ B^{(1)} ∘ E`, then with block types and composition powers,
     e.g. `Ψ ∘ ((B_F ∘ B_L^{∘3})^{∘6}) ∘ E`, plus the encoder towers and a layer-pattern table;
   - a definition of every layer: embedding, norms, each block type as residual sub-layers `(id + g)` with its
     equations and parameter shapes, and the output head;
   - neurons (pre-activation value and active/inactive state) and attention heads (attention pattern);
   - the architecture tuple and parameter set θ, with `|θ| = Σ n_c |θ_c| + |θ_rest|`;
   - output representations (logits, next-token distribution, greedy prediction, causality);
   - the prefix/suffix decomposition `f = G_{ℓ+1} ∘ B^{(ℓ)} ∘ F_{ℓ-1}` and the unrolled residual stream;
   - the local linear (Jacobian) response, and a summary-of-notation table.
3. **Forward Pass in Pseudocode**: commented pseudocode with tensor shapes, for the full model and for each block type.
   The loops run over the layer index ℓ and call `B_c(h; θ_ℓ)`, matching Section 2.
4. **Compact Architecture**: weights of each block type listed once, with repeat count. Includes checkpoint-only
   tensors that transformers never builds (e.g. the Qwen3.5 MTP head).
5. **What Each Layer Does**: a plain-English description of every kind of layer (q_proj, gate_proj, …).
6. **Parameter Count**: by component and by block type.
7. **Appendix**: the original layer-by-layer table with every repetition, and per-layer parameter counts.

## Notes

- `--user "$(id -u):$(id -g)"` ensures output files are owned by you, not root
- `Llama-3.2-1B-Instruct` has exactly the same architecture as `Llama-3.2-1B` (only weights and `eos_token_id` differ), so it has no separate PDF; `Llama-3_2-1B.pdf` names it. Fine-tuned variants like this are listed in `same_arch` in `generate_tex.py`, which checks every tensor name and shape still matches.
