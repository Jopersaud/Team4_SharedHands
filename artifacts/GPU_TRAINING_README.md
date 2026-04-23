# GPU Training Pipeline — ASL Video Dataset

## Other Machine Setup

### 1. Clone / pull the repo
```bash
git pull origin main
```

### 2. Create a venv and install GPU dependencies
```bash
python -m venv .venv_gpu
# Windows:
.venv_gpu\Scripts\activate
# Linux/Mac:
source .venv_gpu/bin/activate

# Install PyTorch with CUDA 12.8 (works with driver 12.9)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining deps
pip install -r artifacts/requirements_gpu.txt
```

### 3. Verify GPU is working
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 5070
```

---

## Pipeline

### Step 1 — Inspect the dataset structure (run before extraction)
```bash
python artifacts/inspect_dataset.py --data_dir /path/to/unzipped/dataset
```
This shows the folder tree and counts videos per letter without extracting anything.
If labels aren't detected correctly, report back so we can adjust the label detection logic.

### Step 2 — Extract landmarks from videos
```bash
python artifacts/extract_video_landmarks.py --data_dir /path/to/unzipped/dataset
```
- Processes videos in parallel (uses all CPU cores)
- Output: `artifacts/video_landmarks/{LETTER}/*.npy`  (each file: 30×63 float32)
- **Resumable** — safe to stop and restart, already-extracted files are skipped
- For 300GB of video this may take several hours depending on CPU

Optional flags:
```bash
--workers 16        # increase parallel workers (default: min(12, cpu_count))
--out_dir custom/   # custom output path
```

### Step 3 — Train the transformer on GPU
```bash
python artifacts/train_transformer_gpu.py
```
- Requires CUDA GPU
- Uses mixed precision (bfloat16) for ~2x speed on RTX 5070
- Trains up to 60 epochs with early stopping (patience=10)
- Saves best checkpoint to `artifacts/asl_transformer.pt`
- Auto-exports ONNX to `public/asl_transformer/model.onnx`

Optional flags:
```bash
--epochs 100
--batch_size 512    # increase if VRAM allows
--lr 3e-4
--patience 15
```

### Step 4 — Copy the ONNX model back to the frontend repo
```bash
# The ONNX model at public/asl_transformer/model.onnx
# needs to be committed and deployed with the React app
git add public/asl_transformer/model.onnx
git commit -m "Add retrained transformer ONNX model"
```

---

## Expected performance
- Extraction: ~1-5 seconds per video (CPU-bound, parallelized)
- Training: ~2-5 minutes per epoch on RTX 5070 depending on dataset size
- Final model accuracy: expect 90-98% on val set with sufficient data

## Troubleshooting
- **CUDA out of memory**: reduce `--batch_size` (try 128 or 64)
- **No videos found**: run `inspect_dataset.py` first to see what structure was detected
- **Low accuracy**: check that extraction found hands (look for zero-filled sequences)
