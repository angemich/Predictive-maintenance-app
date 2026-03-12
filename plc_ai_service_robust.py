from __future__ import annotations

import os
import argparse
from flask import Flask, request, jsonify, Response

from inference_models_robust import Predictor, FEATURES

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

predictor = None


def pick_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0] if paths else None


def build_predictor(stage1_model=None, stage1_scaler=None, stage2_model=None, stage2_scaler=None):
    if stage1_model is None:
        stage1_model = pick_existing(
            os.path.join(BASE, "stage1_scaled_model.ts.pt"),
            os.path.join(BASE, "stage1_torch_model.ts.pt"),
            os.path.join(BASE, "binary_failure_model.pkl"),
            os.path.join(BASE, "binary_failure_model.json"),
        )

    if stage1_scaler is None:
        default_scaler = os.path.join(BASE, "stage1_scaler.npz")
        stage1_scaler = default_scaler if os.path.exists(default_scaler) else None

    if stage2_model is None:
        stage2_model = pick_existing(
            os.path.join(BASE, "failure_type_model.pkl"),
            os.path.join(BASE, "failure_type_model.json"),
        )

    return Predictor(
        stage1_model=stage1_model,
        stage1_scaler=stage1_scaler,
        stage2_model=stage2_model,
        stage2_scaler=stage2_scaler,
    )


@app.get("/health")
def health():
    global predictor

    if predictor is None:
        status = {
            "service": "ok",
            "ready": False,
            "stage1_loaded": False,
            "stage1_kind": None,
            "stage2_loaded": False,
            "stage2_kind": None,
            "warnings": ["Predictor not initialized"],
        }
        return jsonify(status), 503

    status = {
        "service": "ok",
        "ready": predictor.ready,
        "stage1_loaded": predictor.stage1_loaded,
        "stage1_kind": predictor.stage1_kind,
        "stage2_loaded": predictor.stage2_loaded,
        "stage2_kind": predictor.stage2_kind,
        "warnings": getattr(predictor, "warnings", []),
    }
    return (jsonify(status), 200) if predictor.ready else (jsonify(status), 503)


@app.post("/predict")
def predict_json():
    global predictor

    if predictor is None or not predictor.ready:
        return jsonify({"error": "Predictor not ready"}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    try:
        result = predictor.predict_one(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/predict_simple")
def predict_simple():
    global predictor

    if predictor is None or not predictor.ready:
        return Response("ERR: Predictor not ready\n", status=503, mimetype="text/plain")

    try:
        sample = {}
        for f in FEATURES:
            if f not in request.args:
                return Response(f"ERR: missing {f}\n", status=400, mimetype="text/plain")
            sample[f] = float(request.args[f])

        result = predictor.predict_one(sample)

        failure = int(result["machine_failure"])
        prob = float(result["failure_probability"])
        ftype = result["failure_type"]
        ftype_out = int(ftype) if ftype is not None else -1

        return Response(f"{failure},{prob:.6f},{ftype_out}\n", status=200, mimetype="text/plain")

    except Exception as e:
        return Response(f"ERR: {str(e)}\n", status=500, mimetype="text/plain")


def main():
    global predictor

    parser = argparse.ArgumentParser(description="Run PLC AI inference service.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--stage1-model", default=None, help="Path to Stage 1 model")
    parser.add_argument("--stage1-scaler", default=None, help="Path to Stage 1 scaler")
    parser.add_argument("--stage2-model", default=None, help="Path to Stage 2 model")
    parser.add_argument("--stage2-scaler", default=None, help="Path to Stage 2 scaler")

    args = parser.parse_args()

    predictor = build_predictor(
        stage1_model=args.stage1_model,
        stage1_scaler=args.stage1_scaler,
        stage2_model=args.stage2_model,
        stage2_scaler=args.stage2_scaler,
    )

    print("Starting PLC AI service...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Stage 1 model: {args.stage1_model or 'auto-detect'}")
    print(f"Stage 2 model: {args.stage2_model or 'auto-detect'}")
    print(f"Ready: {predictor.ready}")
    print(f"Stage 1 loaded: {predictor.stage1_loaded} ({predictor.stage1_kind})")
    print(f"Stage 2 loaded: {predictor.stage2_loaded} ({predictor.stage2_kind})")

    if predictor.warnings:
        print("Warnings:")
        for w in predictor.warnings:
            print(f" - {w}")

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()