import time
import argparse
from typing import Dict, Any

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURES = [
    "Air_temperature__K_",
    "Process_temperature__K_",
    "Rotational_speed__rpm_",
    "Torque__Nm_",
    "Tool_wear__min_",
    "Type_L",
    "Type_M",
]

np.random.seed(42)
SAMPLE_INTERVAL = 1.0


def wait_for_service(health_url: str, timeout_s: float = 15.0) -> None:
    t0 = time.time()
    last_err = None

    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                j = r.json()
                if j.get("ready") is True:
                    return
                last_err = f"Health says not ready yet: {j}"
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:
            last_err = str(e)

        time.sleep(0.5)

    raise RuntimeError(f"Service not ready after {timeout_s}s. Last error: {last_err}")


def row_to_sample(row: pd.Series) -> Dict[str, Any]:
    air = float(row["Air temperature [K]"])
    proc = float(row["Process temperature [K]"])
    speed = float(row["Rotational speed [rpm]"])
    torque = float(row["Torque [Nm]"])
    wear = float(row["Tool wear [min]"])
    t = row["Type"]

    if t == "L":
        type_l, type_m = 1.0, 0.0
    elif t == "M":
        type_l, type_m = 0.0, 1.0
    else:
        type_l, type_m = 0.0, 0.0

    return {
        "Air_temperature__K_": air,
        "Process_temperature__K_": proc,
        "Rotational_speed__rpm_": speed,
        "Torque__Nm_": torque,
        "Tool_wear__min_": wear,
        "Type_L": type_l,
        "Type_M": type_m,
    }


def call_model(server_url: str, sample: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    last = None
    for k in range(max_retries):
        try:
            r = requests.post(server_url, json=sample, timeout=2)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last = str(e)
        time.sleep(0.2 * (2 ** k))
    raise RuntimeError(f"Model call failed after {max_retries} retries: {last}")


def get_alarm_level(prob: float) -> str:
    if prob < 0.20:
        return "GREEN"
    if prob < 0.90:
        return "YELLOW"
    return "RED"


def run_digital_twin(
    server_base: str,
    data_path: str,
    output_image: str = "digital_twin_csv_dashboard.png",
    timeout_s: float = 15.0,
    max_rows: int | None = None,
    show_plot: bool = True,
):
    server_url = server_base.rstrip("/") + "/predict"
    health_url = server_base.rstrip("/") + "/health"

    print("Waiting for PLC AI service...")
    wait_for_service(health_url=health_url, timeout_s=timeout_s)
    print("Service is ready.")

    df = pd.read_csv(data_path)
    print(f"Dataset loaded with {len(df)} samples from: {data_path}")

    time_log, prob_log, pred_log, true_log, wear_log, torque_log, alarm_log = ([] for _ in range(7))

    for t, (_, row) in enumerate(df.iterrows(), start=1):
        if max_rows is not None and t > max_rows:
            break

        sample = row_to_sample(row)
        data = call_model(server_url=server_url, sample=sample)

        failure = int(data["machine_failure"])
        prob = float(data["failure_probability"])
        ftype_code = data.get("failure_type_code") or "NONE"

        true_failure = int(row["Machine failure"])
        alarm = get_alarm_level(prob)

        time_log.append(t)
        prob_log.append(prob)
        pred_log.append(failure)
        true_log.append(true_failure)
        wear_log.append(sample["Tool_wear__min_"])
        torque_log.append(sample["Torque__Nm_"])
        alarm_log.append(alarm)
        time.sleep(SAMPLE_INTERVAL)
        print(
            f"[t={t:>3}] Wear={sample['Tool_wear__min_']:.1f} | "
            f"Torque={sample['Torque__Nm_']:.1f} | "
            f"Prob={prob:6.3f} | Pred={failure} | True={true_failure} | "
            f"Type={ftype_code} | Alarm={alarm}",
            flush=True,
        )


    
    time_arr = np.array(time_log)
    prob_arr = np.array(prob_log)
    pred_arr = np.array(pred_log)
    true_arr = np.array(true_log)
    wear_arr = np.array(wear_log)
    torque_arr = np.array(torque_log)

    color_map = {"GREEN": "tab:green", "YELLOW": "gold", "RED": "tab:red"}
    colors = [color_map[a] for a in alarm_log]

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax[0].scatter(time_arr, prob_arr, c=colors, s=20, label="P(failure)")
    ax[0].plot(time_arr, prob_arr, alpha=0.2)
    ax[0].axhline(0.20, color="green", linestyle="--", label="Green/Yellow boundary (0.20)")
    ax[0].axhline(0.90, color="red", linestyle="--", label="Yellow/Red boundary (0.90)")
    ax[0].set_ylabel("Failure probability")
    ax[0].set_title("AI Failure Probability with Industrial Alarm Levels")
    ax[0].legend(loc="upper right")
    ax[0].grid(True, alpha=0.2)

    ax[1].step(time_arr, true_arr, where="post", label="True failure (CSV)", linewidth=2)
    ax[1].step(time_arr, pred_arr, where="post", label="Predicted failure", linestyle="--", linewidth=2)
    ax[1].set_ylabel("Failure flag")
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(["No fail", "Fail"])
    ax[1].set_title("True vs Predicted Failure Flags")
    ax[1].legend(loc="upper right")
    ax[1].grid(True, alpha=0.2)

    ax[2].plot(time_arr, wear_arr, label="Tool wear [min]")
    ax[2].plot(time_arr, torque_arr, label="Torque [Nm]", alpha=0.7)
    ax[2].set_xlabel("Time step (row index)")
    ax[2].set_ylabel("Value")
    ax[2].set_title("Physical State (from CSV with noise)")
    ax[2].legend(loc="upper right")
    ax[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Saved dashboard image: {output_image}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "output_image": output_image,
        "samples_processed": len(time_log),
    }


def main():
    parser = argparse.ArgumentParser(description="Run digital twin simulation from CSV.")
    parser.add_argument("--server-base", default="http://127.0.0.1:5000", help="Base URL of PLC AI service")
    parser.add_argument("--data-path", default="ai4i2020_Test.csv", help="Path to CSV dataset")
    parser.add_argument("--output-image", default="digital_twin_csv_dashboard.png", help="Path to save dashboard image")
    parser.add_argument("--timeout", type=float, default=15.0, help="Service readiness timeout in seconds")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows to process")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib window")

    args = parser.parse_args()

    run_digital_twin(
        server_base=args.server_base,
        data_path=args.data_path,
        output_image=args.output_image,
        timeout_s=args.timeout,
        max_rows=args.max_rows,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()