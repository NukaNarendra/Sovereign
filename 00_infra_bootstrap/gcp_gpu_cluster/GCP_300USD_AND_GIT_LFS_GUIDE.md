==============================================================================SOVEREIGN-PROB-7B: GIT LFS + GOOGLE CLOUD ($300 CREDIT) MASTER RUNBOOKFile: 00_infra_bootstrap/gcp_gpu_cluster/GCP_300USD_AND_GIT_LFS_GUIDE.mdPurpose: How to safely push your codebase & Parquet shards to a Private GitHubrepository with Git LFS, debug Stage 3 locally for $0.00, and trainon Google Cloud ($300 Credit = ~375 Hours on NVIDIA L4 24GB GPU).==============================================================================1. Principal Architect Verdict: Why Your New Plan Is 10x Better (With 2 Critical Rules!)Using GitHub + Git LFS and your $300 Google Cloud (GCP) credits is a much better and safer strategy than AMD Cloud for three reasons:7.5x More GPU Hours: On Google Cloud, a g2-standard-8 instance with 1x NVIDIA L4 (24GB VRAM, Ada Lovelace) costs ~$0.80/hr On-Demand (or ~$0.28/hr Spot). Your $300 GCP credit gives you 375 to 1,000+ GPU hours (compared to only 50 hours on AMD!).CUDA Ecosystem Stability: NVIDIA CUDA (torch.cuda, bitsandbytes 4-bit NF4 QLoRA, TRL, peft, and llama.cpp) is 100% native on Google Cloud's Deep Learning VM images—zero experimental ROCm 10.1 driver quirks.Instant git clone Deployment: By pushing your code and the 4,000-row Parquet warehouse (~6.5 MB total!) to GitHub first, you can spin up a GCP GPU VM, run git clone, train, and delete the VM in minutes.Two Golden Rules Before You Push to GitHub:Rule 1 — Make the GitHub Repo PRIVATE (Until We Audit Every File Together):
Keep your startup's trading logic in a Private GitHub repo (or use a Personal Access Token / SSH key) so random bots on the internet cannot steal your Polymarket alpha.Rule 2 — Track .parquet in Git LFS, But DO NOT Push 15 GB .safetensors to GitHub LFS!Why? GitHub's free tier gives you only 1.0 GB of free Git LFS storage.Your 9 Parquet dataset shards (data/processed_warehouse/final_splits/*/*.parquet) are only ~6.6 MB total—they fit in Git LFS easily (0.6% of your 1 GB limit)!However, a full merged 7B model (.safetensors / .gguf) is 4.5 GB to 15.2 GB. If you push 15 GB to GitHub LFS, GitHub will immediately lock your account's LFS quota.The Pro Solution: Store code + .parquet shards in GitHub LFS, and push the heavy trained .safetensors / .gguf model weights to a free Private Hugging Face Model Repo (huggingface.co/new, which gives you UNLIMITED free Git LFS storage for model weights!) or a Google Cloud Storage (gs://) bucket.2. Step-by-Step: Configure Git + Git LFS in Your Local Ubuntu WSL2 TerminalRun this block in your local Ubuntu WSL2 terminal inside /mnt/c/Narendra/python_files/projects/Sovereign-Prob-7B:cd /mnt/c/Narendra/python_files/projects/Sovereign-Prob-7B

# 1. Install Git LFS in Ubuntu WSL2
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install

# 2. Initialize Git repository (if not already initialized)
git init
git branch -M main

# 3. Create .gitattributes so all Parquet datasets are tracked cleanly by Git LFS
cat << 'EOF' > .gitattributes
*.parquet filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
EOF

# 4. Create a strict .gitignore so secrets (.env) and multi-GB base model caches
#    never bloat your GitHub push or leak API keys
cat << 'EOF' > .gitignore
# Secrets & Keys (NEVER COMMIT)
.env
*.pem
*.key
id_ed25519*
00_infra_bootstrap/secrets_vault/*.decrypted*

# Python & Virtual Environments
__pycache__/
*.py[cod]
.venv/
venv/
sovereign_env/

# Heavy Multi-GB Model Binary Weights & Logs (Store on HuggingFace Hub / GCS)
logs/
third_party/
*.safetensors
*.bin
*.pt
*.gguf
.mi300x_training.lock
.gcp_training.lock
EOF

# 5. Stage your entire codebase + the 9 Parquet warehouse splits (~6.6 MB)
git add .gitattributes .gitignore
git add 00_infra_bootstrap/ 01_data_ingestion/ 02_dataset_factory/ 03_llm_training/ 04_mcp_runtime/
git add -f data/processed_warehouse/final_splits/

# 6. Check status and commit
git status
git commit -m "feat: Sovereign-Prob-7B Stage 1-6 pipeline + 4,000-row Parquet warehouse via Git LFS"
Then create a new repository on GitHub (e.g. https://github.com/<your-username>/Sovereign-Prob-7B.git) and push:git remote add origin https://github.com/<your-username>/Sovereign-Prob-7B.git
git push -u origin main
3. Step-by-Step: Launch Your Google Cloud ($300 Credit) NVIDIA L4 GPU VMStep 1: Check / Enable GPU Quota on Google CloudOpen Google Cloud Console.Go to IAM & Admin $\rightarrow$ Quotas & System Limits.In the filter box, type: NVIDIA L4 GPUs (or GPUs (all regions)).Note for $300 Free Trial accounts: If your account says "Free Trial accounts cannot create GPUs", click the "Upgrade / Activate Full Account" button at the top of GCP Console. Important: Activating your full account does NOT delete your $300 free credit! You keep the entire $300 credit for the remaining 90 days, and it unlocks NVIDIA L4 (g2-standard-8) and NVIDIA T4 (n1-standard-8) GPUs immediately.Step 2: Launch a Deep Learning VM (g2-standard-8 with 1x NVIDIA L4 24GB VRAM)Either click Compute Engine $\rightarrow$ VM Instances $\rightarrow$ Create Instance in the browser, or run this single gcloud command in Google Cloud Shell:gcloud compute instances create sovereign-prob-7b-l4 \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-2-4-cu124-ubuntu-2204-nvidia-550 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=150GB \
    --boot-disk-type=pd-ssd
Step 3: Clone Your GitHub LFS Repo on the GCP GPU VM & Run Training!SSH into your GCP instance (gcloud compute ssh sovereign-prob-7b-l4 --zone=us-central1-a), then run:# 1. Start persistent tmux session
tmux new -s sovereign_gcp

# 2. Clone your GitHub repo (automatically pulls the 9 Parquet shards via Git LFS!)
git lfs install
git clone https://github.com/<your-username>/Sovereign-Prob-7B.git
cd Sovereign-Prob-7B

# 3. Install required Python packages into the GCP PyTorch CUDA environment
pip install -q polars pyarrow pydantic httpx trl peft bitsandbytes accelerate transformers datasets

# 4. Run Preflight First (CONFIRM_TRAINING=NO — verifies CUDA, L4 24GB VRAM, and Parquet shards)
chmod +x 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh
CONFIRM_TRAINING=NO bash 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh

# 5. Once Preflight passes, run the real training pipeline!
CONFIRM_TRAINING=YES bash 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh
