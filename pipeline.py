"""
Minit-ML pipeline orchestrator.
Manages the two-stage cascade inference, routing, and distillation triggers.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("pipeline")


@dataclass
class ScanResult:
    file_path: str
    file_hash: str
    file_type: str

    nsfw_prob: float = 0.0
    nsfw_label: str = "n/a"
    nsfw_stage2_checked: bool = False

    malware_prob: float = 0.0
    malware_label: str = "n/a"
    malware_stage2_checked: bool = False

    stage_used: int = 1
    latency_ms: float = 0.0
    blocked: bool = False

    @property
    def verdict(self) -> int:
        if self.file_type == "image":
            if self.nsfw_label == "safe":
                return 0
            elif self.nsfw_label == "nsfw":
                return 1
            elif self.nsfw_label == "suspicious":
                return 2
        elif self.file_type == "pe":
            if self.malware_label == "safe":
                return 0
            elif self.malware_label == "malware":
                return 1
            elif self.malware_label == "suspicious":
                return 2
        return 0


class CascadePipeline:
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self._nsfw_detector = None
        self._pe_detector = None
        self._nudenet_detector = None
        self._ember_detector = None
        self._nsfw_distiller = None
        self._malware_distiller = None

        self.block_threshold_nsfw = config.get("block_threshold_nsfw", 0.7)
        self.block_threshold_malware = config.get("block_threshold_malware", 0.8)
        self.stage2_enabled = config.get("stage2_enabled", True)

    def _get_nsfw(self):
        if self._nsfw_detector is None:
            from src.stage1.clip_nsfw import CLIPNSFWDetector
            model_path = os.path.join(os.path.dirname(__file__), "models/clip_nsfw_head.pth")
            self._nsfw_detector = CLIPNSFWDetector(
                model_path=model_path if os.path.exists(model_path) else None,
            )
        return self._nsfw_detector

    def _get_pe(self):
        if self._pe_detector is None:
            from src.stage1.pe_lite import PELiteDetector
            model_path = os.path.join(os.path.dirname(__file__), "models/pe_lite_xgb.json")
            self._pe_detector = PELiteDetector(
                model_path=model_path if os.path.exists(model_path) else None,
            )
        return self._pe_detector

    def _get_nudenet(self):
        if self._nudenet_detector is None:
            from src.stage2.nudenet_nsfw import NudeNetDetector
            self._nudenet_detector = NudeNetDetector()
        return self._nudenet_detector

    def _get_ember(self):
        if self._ember_detector is None:
            from src.stage2.ember_malware import EmberMalwareDetector
            model_path = os.path.join(os.path.dirname(__file__), "models/ember_lgbm.txt")
            self._ember_detector = EmberMalwareDetector(
                model_path=model_path if os.path.exists(model_path) else None,
            )
        return self._ember_detector

    def _get_nsfw_distiller(self):
        if self._nsfw_distiller is None:
            from src.distillation.nsfw_distill import NSFWDistiller
            buffer_path = os.path.join(os.path.dirname(__file__), "data/nsfw_distill_buffer.jsonl")
            self._nsfw_distiller = NSFWDistiller(buffer_path=buffer_path)
        return self._nsfw_distiller

    def _get_malware_distiller(self):
        if self._malware_distiller is None:
            from src.distillation.malware_distill import MalwareDistiller
            buffer_path = os.path.join(os.path.dirname(__file__), "data/malware_distill_buffer.jsonl")
            self._malware_distiller = MalwareDistiller(buffer_path=buffer_path)
        return self._malware_distiller

    @staticmethod
    def detect_file_type(file_path: str, file_bytes: bytes = None) -> str:
        if file_bytes is None:
            with open(file_path, 'rb') as f:
                file_bytes = f.read(4096)

        if file_bytes[:2] == b'MZ':
            return "pe"

        image_signatures = [
            b'\xff\xd8\xff',
            b'\x89PNG\r\n\x1a\n',
            b'GIF87a',
            b'GIF89a',
            b'RIFF',
            b'BM',
        ]
        for sig in image_signatures:
            if file_bytes[:len(sig)] == sig:
                return "image"

        return "unknown"

    def scan(self, file_path: str) -> ScanResult:
        start = time.time()

        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        file_type = self.detect_file_type(file_path, file_bytes)

        import hashlib
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        result = ScanResult(
            file_path=file_path,
            file_hash=file_hash,
            file_type=file_type,
        )

        if file_type == "image":
            nsfw = self._get_nsfw()
            stage1_result = nsfw.predict(file_path)
            result.nsfw_prob = stage1_result["nsfw_prob"]
            result.nsfw_label = stage1_result["label"]

            if stage1_result["label"] == "safe":
                result.blocked = False
                result.stage_used = 1
            elif stage1_result["label"] == "nsfw":
                result.blocked = True
                result.stage_used = 1
            else:
                result.stage_used = 1
                if self.stage2_enabled:
                    try:
                        nudenet = self._get_nudenet()
                        stage2_result = nudenet.predict(file_path)
                        result.nsfw_stage2_checked = True
                        result.stage_used = 2

                        self._get_nsfw_distiller().collect_sample(
                            embedding=stage1_result["embedding"],
                            stage1_prob=stage1_result["nsfw_prob"],
                            stage2_result=stage2_result,
                        )

                        if stage2_result["is_nsfw"]:
                            result.nsfw_prob = max(result.nsfw_prob, stage2_result["nsfw_score"])
                            result.nsfw_label = "nsfw"
                            result.blocked = True
                        else:
                            result.nsfw_label = "safe"
                            result.blocked = False
                    except Exception as e:
                        logger.warning(f"Stage 2 NSFW failed: {e}")
                        result.blocked = False
                else:
                    result.blocked = False

        elif file_type == "pe":
            pe = self._get_pe()
            stage1_result = pe.predict(file_path=file_path, file_bytes=file_bytes)
            result.malware_prob = stage1_result["malware_prob"]
            result.malware_label = stage1_result["label"]

            if stage1_result["label"] == "safe":
                result.blocked = False
                result.stage_used = 1
            elif stage1_result["label"] == "malware":
                result.blocked = True
                result.stage_used = 1
            else:
                result.stage_used = 1
                if self.stage2_enabled:
                    try:
                        ember = self._get_ember()
                        stage2_result = ember.predict(file_path=file_path, file_bytes=file_bytes)
                        result.malware_stage2_checked = True
                        result.stage_used = 2

                        self._get_malware_distiller().collect_sample(
                            pe_features=stage1_result["features"],
                            stage1_prob=stage1_result["malware_prob"],
                            stage2_result=stage2_result,
                            file_hash=file_hash,
                        )

                        if stage2_result["is_malware"]:
                            result.malware_prob = max(result.malware_prob, stage2_result["malware_prob"])
                            result.malware_label = "malware"
                            result.blocked = True
                        else:
                            result.malware_label = "safe"
                            result.blocked = False
                    except Exception as e:
                        logger.warning(f"Stage 2 Malware failed: {e}")
                        result.blocked = False
                else:
                    result.blocked = False

        else:
            result.blocked = False

        result.latency_ms = (time.time() - start) * 1000

        return result

    def run_distillation(self, domain: str = "all") -> dict:
        results = {}

        if domain in ("all", "nsfw"):
            results["nsfw"] = self._get_nsfw_distiller().distill(
                student_model=self._get_nsfw().head,
                output_path=os.path.join(os.path.dirname(__file__), "models/clip_nsfw_head.pth"),
            )
            if results["nsfw"].get("status") == "success":
                self._nsfw_detector = None
                logger.info("NSFW model reloaded.")

        if domain in ("all", "malware"):
            results["malware"] = self._get_malware_distiller().distill(
                original_csv=os.path.join(os.path.dirname(__file__), "data/top_1000_pe_imports.csv"),
                output_model=os.path.join(os.path.dirname(__file__), "models/pe_lite_xgb.json"),
            )
            if results["malware"].get("status") == "success":
                self._pe_detector = None
                logger.info("Malware model reloaded.")

        return results

    def stats(self) -> dict:
        return {
            "nsfw_distill_buffer": self._get_nsfw_distiller().buffer.count,
            "malware_distill_buffer": self._get_malware_distiller().buffer.count,
            "stage2_enabled": self.stage2_enabled,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Minit-ML Cascade Pipeline")
    sub = parser.add_subparsers(dest="command")

    scan_p = sub.add_parser("scan")
    scan_p.add_argument("file")

    distill_p = sub.add_parser("distill")
    distill_p.add_argument("--domain", choices=["nsfw", "malware", "all"], default="all")

    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--port", type=int, default=8099)
    serve_p.add_argument("--host", default="0.0.0.0")

    sub.add_parser("stats")

    args = parser.parse_args()
    pipeline = CascadePipeline()

    if args.command == "scan":
        result = pipeline.scan(args.file)
        print(f"File:       {result.file_path}")
        print(f"Type:       {result.file_type}")
        print(f"Hash:       {result.file_hash[:16]}...")
        if result.file_type == "image":
            print(f"NSFW:       {result.nsfw_prob:.4f} ({result.nsfw_label})")
            print(f"Stage2:     {result.nsfw_stage2_checked}")
        if result.file_type == "pe":
            print(f"Malware:    {result.malware_prob:.4f} ({result.malware_label})")
            print(f"Stage2:     {result.malware_stage2_checked}")
        print(f"Verdict:    {result.verdict} ({'OK' if result.verdict == 0 else 'DANGER' if result.verdict == 1 else 'SUSPICIOUS'})")
        print(f"Stage used: {result.stage_used}")
        print(f"Latency:    {result.latency_ms:.1f}ms")
        print(f"BLOCKED:    {'YES' if result.blocked else 'NO'}")

    elif args.command == "distill":
        results = pipeline.run_distillation(args.domain)
        for domain, stats in results.items():
            print(f"\n[{domain.upper()}]")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    elif args.command == "serve":
        try:
            import uvicorn
            app = create_fastapi_app(pipeline)
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            print("pip install uvicorn fastapi")
            sys.exit(1)

    elif args.command == "stats":
        stats = pipeline.stats()
        for k, v in stats.items():
            print(f"{k}: {v}")

    else:
        parser.print_help()


def verify_in_background(pipeline: CascadePipeline, file_path: str, file_hash: str):
    try:
        orig_stage2 = pipeline.stage2_enabled
        pipeline.stage2_enabled = True
        result = pipeline.scan(file_path)
        pipeline.stage2_enabled = orig_stage2

        results_path = os.path.join(os.path.dirname(__file__), "data/verification_results.jsonl")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        record = {
            "timestamp": time.time(),
            "file_hash": file_hash,
            "verdict": 1 if result.blocked else 0,
            "file_type": result.file_type,
            "nsfw_prob": result.nsfw_prob,
            "malware_prob": result.malware_prob
        }
        with open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"Background verification completed for {file_hash}: verdict={record['verdict']}")
    except Exception as e:
        logger.error(f"Background verification failed for {file_hash}: {e}")
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


def create_fastapi_app(pipeline: CascadePipeline = None):
    from fastapi import FastAPI, UploadFile, File, BackgroundTasks
    import tempfile
    import shutil

    app = FastAPI(title="Minit-ML Content Moderation API")
    pipeline = pipeline or CascadePipeline()

    @app.post("/scan")
    async def scan_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            orig_stage2 = pipeline.stage2_enabled
            pipeline.stage2_enabled = False
            result = pipeline.scan(tmp_path)
            pipeline.stage2_enabled = orig_stage2

            if result.verdict == 2:
                queue_dir = os.path.join(os.path.dirname(__file__), "data/verification_queue")
                os.makedirs(queue_dir, exist_ok=True)
                queue_path = os.path.join(queue_dir, result.file_hash)
                shutil.copy(tmp_path, queue_path)
                background_tasks.add_task(verify_in_background, pipeline, queue_path, result.file_hash)

            return {
                "verdict": result.verdict,
                "file": file.filename,
                "type": result.file_type,
                "nsfw": {"prob": result.nsfw_prob, "label": result.nsfw_label},
                "malware": {"prob": result.malware_prob, "label": result.malware_label},
                "latency_ms": round(result.latency_ms, 1),
                "blocked": result.blocked,
            }
        finally:
            os.unlink(tmp_path)

    @app.post("/verify")
    async def verify_suspicious(file: UploadFile = File(...)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            orig_stage2 = pipeline.stage2_enabled
            pipeline.stage2_enabled = True
            result = pipeline.scan(tmp_path)
            pipeline.stage2_enabled = orig_stage2

            return {
                "verdict": 1 if result.blocked else 0,
                "file": file.filename,
                "type": result.file_type,
                "stage_used": result.stage_used,
                "nsfw": {"prob": result.nsfw_prob, "label": result.nsfw_label, "stage2": result.nsfw_stage2_checked},
                "malware": {"prob": result.malware_prob, "label": result.malware_label, "stage2": result.malware_stage2_checked},
                "latency_ms": round(result.latency_ms, 1),
                "blocked": result.blocked,
            }
        finally:
            os.unlink(tmp_path)

    @app.get("/verify/{file_hash}")
    async def get_verification_result(file_hash: str):
        results_path = os.path.join(os.path.dirname(__file__), "data/verification_results.jsonl")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    if record["file_hash"] == file_hash:
                        return {
                            "status": "completed",
                            "verdict": record["verdict"],
                            "timestamp": record["timestamp"]
                        }

        queue_dir = os.path.join(os.path.dirname(__file__), "data/verification_queue")
        queue_path = os.path.join(queue_dir, file_hash)
        if os.path.exists(queue_path):
            return {"status": "pending"}

        return {"status": "not_found"}

    @app.post("/distill/{domain}")
    async def trigger_distillation(domain: str):
        results = pipeline.run_distillation(domain)
        return results

    @app.get("/stats")
    async def get_stats():
        return pipeline.stats()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    main()
