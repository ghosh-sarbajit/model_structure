# Model Architecture .tex Generation & Compilation

Generate detailed LaTeX architecture diagrams for transformer models, then compile to PDF.

## Required Files

- `sg_modelLoad` — CUDA + Python 3.12 + torch + transformers image
- `generate_tex.py` — loads each model and writes a `.tex` file with full layer-by-layer architecture
- `load_models.py` — utility to verify all models load correctly on GPU
- `Dockerfile.texlive` — TeX Live full image (compile `.tex` → `.pdf`)

## Order of Execution

### 1. Build the ML image
```bash
docker build --label owner=$(whoami) --build-arg OWNER=$(whoami) \
    -t sg_modelload -f sg_modelLoad .
```

### 2. Generate `.tex` files from all models
```bash
docker run --rm --gpus '"device=2"' \
    --user "$(id -u):$(id -g)" \
    -v /opt/dlami/nvme/sarbajit/local_models:/models \
    -v /opt/dlami/nvme/sarbajit/model_structure:/workspace \
    -w /workspace \
    sg_modelload \
    python generate_tex.py
```
Output: `tex_output/*.tex` (one per model)

### 3. Build TeX Live image
```bash
docker build --label owner=$(whoami) --build-arg OWNER=$(whoami) \
    -t texlive-full -f Dockerfile.texlive .
```

### 4. Compile `.tex` → `.pdf`
```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v /opt/dlami/nvme/sarbajit/model_structure:/workdir \
    -w /workdir \
    texlive-full \
    sh -c 'mkdir -p tex_output/pdf && \
           for f in tex_output/*.tex; do \
             pdflatex -interaction=nonstopmode -output-directory=tex_output "$f"; \
           done && \
           mv tex_output/*.pdf tex_output/pdf/'
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

- **Model Overview** — total params, weight dtype
- **Forward Pass Algorithm** — pseudocode with tensor shapes (`B × T × hidden`)
- **Layer-by-Layer Table** — every weight tensor: path, type, shape, dtype, parameter count
- **Parameter Count Estimation** — breakdown by component (Embedding, Attention, MLP, LayerNorm, LM Head)

## Notes

- `--user "$(id -u):$(id -g)"` ensures output files are owned by you, not root
- `--gpus '"device=2"'` selects GPU 2; change the number for a different GPU
- Models are loaded sequentially on GPU and fully analysed
