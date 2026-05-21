.PHONY: setup train-stage1 train-stage2 init run distill-nsfw distill-malware clean

VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

setup:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing core dependencies from requirements.txt..."
	$(PIP) install -r requirements.txt
	@echo "Installing pyyaml..."
	$(PIP) install pyyaml
	@echo "Installing matplotlib..."
	$(PIP) install matplotlib
	@echo "Installing local references (ember, NudeNet)..."
	$(PIP) install -e references/ember
	$(PIP) install -e references/NudeNet
	@echo "Setup completed successfully."

train-stage1:
	@echo "Training Stage 1 Malware model (PE-Lite XGBoost)..."
	$(PYTHON) src/stage1/pe_lite.py train
	@echo "Initializing default CLIP NSFW Head weights..."
	$(PYTHON) -c "import torch; from src.stage1.clip_nsfw import CLIPNSFWHead; import os; os.makedirs('models', exist_ok=True); torch.save(CLIPNSFWHead(input_size=512).state_dict(), 'models/clip_nsfw_head.pth')"

train-stage2:
	@echo "Training Stage 2 Malware model (EMBER LightGBM)..."
	$(PYTHON) -c "import pandas as pd; import lightgbm as lgb; import numpy as np; df = pd.read_csv('data/MalwareData.csv', sep='|'); y = (1 - df['legitimate'].values).astype(np.float32); X = df.drop(columns=['Name', 'md5', 'legitimate']).values.astype(np.float32); dtrain = lgb.Dataset(X, label=y); model = lgb.train({'objective': 'binary', 'metric': 'auc', 'num_leaves': 31, 'learning_rate': 0.05, 'verbose': -1}, dtrain, num_boost_round=100); os.makedirs('models', exist_ok=True); model.save_model('models/ember_lgbm.txt')"

init: setup train-stage1 train-stage2
	@echo "Initialization completed. Ready to serve!"

run:
	@echo "Starting FastAPI server on port 8099..."
	KMP_DUPLICATE_LIB_OK=TRUE $(PYTHON) pipeline.py serve --port 8099 --host 0.0.0.0

distill-nsfw:
	@echo "Triggering NSFW distillation training..."
	KMP_DUPLICATE_LIB_OK=TRUE $(PYTHON) pipeline.py distill --domain nsfw

distill-malware:
	@echo "Triggering Malware distillation training..."
	KMP_DUPLICATE_LIB_OK=TRUE $(PYTHON) pipeline.py distill --domain malware

clean:
	@echo "Cleaning up temporary files, caches, and logs..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f data/benchmark_output.txt data/manual_benchmark.txt
	@echo "Cleanup finished."
