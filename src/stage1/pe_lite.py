"""
Stage 1 PE Malware Detector.
Extracts PE import features and performs binary classification using XGBoost.
"""

import os
import hashlib
import numpy as np
import pefile
import xgboost as xgb
from pathlib import Path
from typing import Optional, Tuple, Dict, List

TOP_IMPORTS = None


def _load_top_imports(csv_path: str = None) -> List[str]:
    global TOP_IMPORTS
    if TOP_IMPORTS is not None:
        return TOP_IMPORTS

    import csv
    csv_path = csv_path or os.path.join(
        os.path.dirname(__file__),
        "../../data/top_1000_pe_imports.csv"
    )
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            cols = [c for c in reader.fieldnames if c not in ('hash', 'malware')]
            TOP_IMPORTS = cols[:1000]
    else:
        TOP_IMPORTS = [
            "GetProcAddress", "LoadLibraryA", "VirtualAlloc", "WriteProcessMemory",
            "CreateRemoteThread", "OpenProcess", "VirtualProtectEx", "ReadProcessMemory",
            "CreateFileA", "WriteFile", "RegSetValueExA", "RegCreateKeyExA",
            "InternetOpenA", "HttpOpenRequestA", "InternetReadFile", "URLDownloadToFileA",
            "WinExec", "ShellExecuteA", "CreateServiceA", "StartServiceA",
            "NtUnmapViewOfSection", "SetWindowsHookExA", "AdjustTokenPrivileges",
            "IsDebuggerPresent", "GetTickCount", "Sleep", "CreateMutexA",
            "CryptEncrypt", "CryptDecrypt", "SetFileAttributesA",
            "FindFirstFileA", "FindNextFileA", "DeleteFileA", "MoveFileA",
            "GetSystemDirectoryA", "GetTempPathA", "GetWindowsDirectoryA",
        ]
    return TOP_IMPORTS


def extract_pe_imports(file_path: str = None, file_bytes: bytes = None) -> np.ndarray:
    top_imports = _load_top_imports()
    features = np.zeros(len(top_imports), dtype=np.float32)
    import_set = set()

    try:
        if file_bytes:
            pe = pefile.PE(data=file_bytes)
        elif file_path:
            pe = pefile.PE(file_path)
        else:
            return features

        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        import_set.add(imp.name.decode('utf-8', errors='ignore'))
        pe.close()
    except Exception:
        return features

    for i, name in enumerate(top_imports):
        if name in import_set:
            features[i] = 1.0

    return features


def extract_ember_features(file_path: str = None, file_bytes: bytes = None) -> Optional[np.ndarray]:
    try:
        from ember.features import PEFeatureExtractor
        extractor = PEFeatureExtractor(feature_version=2)
        if file_bytes is None:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
        return extractor.feature_vector(file_bytes)
    except ImportError:
        print("[EMBER] ember not installed")
        return None


class PELiteDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        suspicious_range: Tuple[float, float] = (0.2, 0.8),
    ):
        self.suspicious_range = suspicious_range
        self.model = None

        if model_path and os.path.exists(model_path):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            print(f"[Stage1-Malware] Loaded XGBoost from {model_path}")
        else:
            print("[Stage1-Malware] No pre-trained model")

    def predict(self, file_path: str = None, file_bytes: bytes = None) -> dict:
        features = extract_pe_imports(file_path, file_bytes)

        if file_bytes is None:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if self.model is None:
            return {
                "malware_prob": 0.5,
                "label": "suspicious",
                "needs_stage2": True,
                "features": features,
                "file_hash": file_hash,
            }

        dmat = xgb.DMatrix(features.reshape(1, -1))
        prob = float(self.model.predict(dmat)[0])

        lo, hi = self.suspicious_range
        if prob < lo:
            label = "safe"
        elif prob > hi:
            label = "malware"
        else:
            label = "suspicious"

        return {
            "malware_prob": prob,
            "label": label,
            "needs_stage2": label == "suspicious",
            "features": features,
            "file_hash": file_hash,
        }

    def predict_from_imports_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if 'hash' in df.columns:
            df = df.drop('hash', axis=1)
        if 'malware' in df.columns:
            df = df.drop('malware', axis=1)

        features = df.values.astype(np.float32)
        if self.model:
            dmat = xgb.DMatrix(features)
            preds = self.model.predict(dmat)
        else:
            preds = np.full(len(features), 0.5)

        return features, preds


def train_pe_lite(
    train_csv: str,
    output_model: str = "models/pe_lite_xgb.json",
    test_size: float = 0.2,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    df = pd.read_csv(train_csv)
    y = df['malware'].values
    X = df.drop(['hash', 'malware'], axis=1, errors='ignore').values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': sum(y == 0) / max(sum(y == 1), 1),
    }

    model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, 'test')])

    preds = (model.predict(dtest) > 0.5).astype(int)
    print(classification_report(y_test, preds, target_names=['benign', 'malware']))

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    model.save_model(output_model)
    print(f"[Stage1-Malware] Model saved to {output_model}")

    return model


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        csv = sys.argv[2] if len(sys.argv) > 2 else "data/top_1000_pe_imports.csv"
        train_pe_lite(csv)
    elif len(sys.argv) > 1:
        detector = PELiteDetector("models/pe_lite_xgb.json")
        result = detector.predict(sys.argv[1])
        print(f"File: {sys.argv[1]}")
        print(f"  Malware prob: {result['malware_prob']:.4f}")
        print(f"  Label: {result['label']}")
        print(f"  Needs Stage 2: {result['needs_stage2']}")
    else:
        print("Usage: python pe_lite.py <pe_file> | python pe_lite.py train <csv>")
