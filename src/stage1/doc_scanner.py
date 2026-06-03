"""
Stage 1 Document / Code Scanner.
Covers PDF, Office (PPTX/DOCX/XLSX/PPT/DOC/XLS), and source code files.
Extracts structural heuristic features → XGBoost binary classification.
"""

import os
import math
import zipfile
import numpy as np
import xgboost as xgb
from typing import Optional, Tuple


CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".sh", ".bash", ".ps1", ".rb", ".php",
    ".java", ".c", ".cpp", ".go", ".rs", ".lua", ".vbs", ".bat",
    ".cmd", ".pl", ".cs", ".swift", ".kt",
}

SUSPICIOUS_CODE_PATTERNS = [
    b"eval(", b"exec(", b"__import__", b"base64.b64decode",
    b"subprocess", b"os.system", b"os.popen", b"shell=True",
    b"powershell", b"invoke-expression", b"iex(", b"wget ", b"curl ",
    b"chmod +x", b"nc -e", b"/bin/sh", b"cmd.exe",
    b"CreateObject", b"WScript.Shell", b"ActiveXObject",
]

SUSPICIOUS_PDF_KEYWORDS = [
    b"/JS", b"/JavaScript", b"/AA", b"/OpenAction",
    b"/Launch", b"/EmbeddedFile", b"/RichMedia",
    b"/XObject", b"/ObjStm", b"/URI",
]

SUSPICIOUS_OFFICE_PARTS = [
    "word/vbaProject.bin",
    "xl/vbaProject.bin",
    "ppt/vbaProject.bin",
    "xl/externalLinks/",
    "word/externalLinks/",
    "_rels/",
]


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))


def extract_pdf_features(file_bytes: bytes) -> np.ndarray:
    features = np.zeros(12, dtype=np.float32)
    features[0] = _entropy(file_bytes)
    features[1] = len(file_bytes)

    for i, kw in enumerate(SUSPICIOUS_PDF_KEYWORDS):
        features[2 + i] = float(file_bytes.count(kw))

    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(file_bytes), strict=False)
        features[11] = len(reader.pages)
    except Exception:
        pass

    return features


def extract_office_features(file_bytes: bytes) -> np.ndarray:
    features = np.zeros(8, dtype=np.float32)
    features[0] = _entropy(file_bytes)
    features[1] = len(file_bytes)

    try:
        import io
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = zf.namelist()
            features[2] = len(names)
            for part in SUSPICIOUS_OFFICE_PARTS:
                if any(n.startswith(part) or n == part for n in names):
                    features[3] += 1
            has_vba = any("vbaProject" in n for n in names)
            features[4] = float(has_vba)
            ext_links = sum(1 for n in names if "externalLinks" in n)
            features[5] = float(ext_links)
    except Exception:
        pass

    return features


def extract_ole_features(file_bytes: bytes) -> np.ndarray:
    features = np.zeros(6, dtype=np.float32)
    features[0] = _entropy(file_bytes)
    features[1] = len(file_bytes)

    try:
        import olefile
        import io
        ole = olefile.OleFileIO(io.BytesIO(file_bytes))
        streams = ole.listdir()
        features[2] = len(streams)
        has_vba = ole.exists("Macros") or ole.exists("VBA") or any(
            "vba" in "/".join(s).lower() for s in streams
        )
        features[3] = float(has_vba)
        ole.close()
    except Exception:
        pass

    return features


def extract_code_features(file_bytes: bytes) -> np.ndarray:
    features = np.zeros(4 + len(SUSPICIOUS_CODE_PATTERNS), dtype=np.float32)
    features[0] = _entropy(file_bytes)
    features[1] = len(file_bytes)
    features[2] = file_bytes.count(b"\n")

    lower = file_bytes.lower()
    for i, pat in enumerate(SUSPICIOUS_CODE_PATTERNS):
        features[4 + i] = float(lower.count(pat))

    return features


class DocScanner:
    def __init__(
        self,
        model_path: Optional[str] = None,
        suspicious_range: Tuple[float, float] = (0.3, 0.7),
    ):
        self.suspicious_range = suspicious_range
        self.model = None

        if model_path and os.path.exists(model_path):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            print(f"[Stage1-Doc] Loaded XGBoost from {model_path}")
        else:
            print("[Stage1-Doc] No pre-trained model — heuristic mode")

    def predict(self, file_path: str, file_bytes: bytes, file_type: str) -> dict:
        if file_type == "pdf":
            features = extract_pdf_features(file_bytes)
        elif file_type == "office":
            features = extract_office_features(file_bytes)
        elif file_type == "ole":
            features = extract_ole_features(file_bytes)
        elif file_type == "code":
            features = extract_code_features(file_bytes)
        else:
            features = np.zeros(8, dtype=np.float32)

        if self.model is None:
            # 모델 없으면 간단한 휴리스틱으로 점수 추정
            prob = self._heuristic_score(features, file_type)
        else:
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
        }

    def _heuristic_score(self, features: np.ndarray, file_type: str) -> float:
        if file_type == "pdf":
            # features[2..11] = suspicious keyword counts
            hits = float(np.sum(features[2:11] > 0))
            return min(hits / 5.0, 1.0)
        elif file_type in ("office", "ole"):
            has_vba = features[3] if len(features) > 3 else 0.0
            has_ext = features[5] if len(features) > 5 else 0.0
            return min((has_vba * 0.5 + has_ext * 0.3), 1.0)
        elif file_type == "code":
            hits = float(np.sum(features[4:] > 0))
            return min(hits / 4.0, 1.0)
        return 0.1
