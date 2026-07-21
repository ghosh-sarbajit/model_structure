============================================
 Model Architecture .tex Generation + Compilation
============================================

--- Required Files ---

1. sg_modelLoad          - Dockerfile for ML image (CUDA + Python 3.12 + torch + transformers)
2. generate_tex.py       - Python script that loads each model and writes a .tex file
3. Dockerfile.texlive    - Dockerfile for TeX Live full image (compiles .tex -> .pdf)
4. COMMANDS.md           - Quick reference for all docker commands

--- Order of Execution ---

Step 1: Build the ML image (if not already built)
  docker build --label owner=$(whoami) --build-arg OWNER=$(whoami) \
      -t sg_modelload -f sg_modelLoad .

Step 2: Generate .tex files from all models
  docker run --rm --gpus '"device=2"' \
      --user "$(id -u):$(id -g)" \
      -v /opt/dlami/nvme/sarbajit/local_models:/models \
      -v /opt/dlami/nvme/sarbajit/DockerExperiment:/workspace \
      -w /workspace \
      sg_modelload \
      python generate_tex.py

  Output: tex_output/*.tex  (one per model)

Step 3: Build TeX Live image (if not already built)
  docker build --label owner=$(whoami) --build-arg OWNER=$(whoami) \
      -t texlive-full -f Dockerfile.texlive .

Step 4: Compile .tex files to .pdf
  docker run --rm \
      --user "$(id -u):$(id -g)" \
      -v /opt/dlami/nvme/sarbajit/DockerExperiment:/workdir \
      -w /workdir \
      texlive-full \
      sh -c 'mkdir -p tex_output/pdf && \
             for f in tex_output/*.tex; do \
               pdflatex -interaction=nonstopmode -output-directory=tex_output "$f"; \
             done && \
             mv tex_output/*.pdf tex_output/pdf/'

  Output: tex_output/pdf/*.pdf

--- Output Structure ---

tex_output/
  ├── *.tex           (one LaTeX file per model)
  ├── *.pdf           (compiled PDFs before move)
  ├── *.log           (LaTeX logs)
  ├── *.aux           (LaTeX aux files)
  └── pdf/
      └── *.pdf       (final compiled PDFs)

--- Notes ---

- --user "$(id -u):$(id -g)" ensures files are owned by you, not root.
- --gpus '"device=2"' selects GPU 2. Change the number for a different GPU.
- The generate step loads each model sequentially on GPU and extracts
  every layer's weight shapes, types, dtypes, and parameter counts.
- Each .tex contains: model overview, forward-pass algorithm in
  algorithmic pseudocode, a full layer-by-layer weight table, and
  a parameter-count breakdown.
