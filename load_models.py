import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_paths = [
    "/models/memorization/Llama-3.2-1B",
    "/models/memorization/Qwen3-0.6B",
    "/models/memorization/Qwen3.5-0.8B",
    "/models/memorization/gemma-4-E2B",
    "/models/sayak_da_work/Llama-3.1-8B-Instruct",
    "/models/sayak_da_work/Llama-3.2-1B-Instruct",
    "/models/sayak_da_work/Qwen2.5-0.5B-Instruct",
    "/models/sayak_da_work/Qwen3.5-9B",
    "/models/gemma-3-12b-it",
    "/models/gpt-oss-20b",
    "/models/Llama-3.1-8B-Instruct",
]

for path in model_paths:
    name = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"Loading: {name} ({path})")
    print('='*60)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = model.to("cuda")

        n_gpu = sum(p.device.type == "cuda" for p in model.parameters())
        params_gb = model.num_parameters() * 2 / 1024**3
        print(f"  Tokenizer: {type(tokenizer).__name__}")
        print(f"  Model: {type(model).__name__}")
        print(f"  Params: {model.num_parameters():,} (~{params_gb:.1f} GB)")
        print(f"  GPU params: {n_gpu}/{n_gpu}")
        del model, tokenizer
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  GPU memory after cleanup: {mem:.2f} GB")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nDone.")
