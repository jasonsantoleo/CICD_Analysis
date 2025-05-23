#!/usr/bin/env python3
# generate_logs.py

import os
import json
import random
import argparse
import socket
import getpass
import threading
from datetime import datetime, timedelta

STAGES = ["checkout", "install", "build", "test", "deploy"]

# Intermediate message templates for each stage
INTERMEDIATE = {
    "checkout": [
        "Fetching remote refs",
        "Validating commit object",
        "Checking out branch"
    ],
    "install": [
        "Resolving dependencies",
        "Downloading packages",
        "Verifying checksums"
    ],
    "build": [
        "Starting compiler",
        "Compiling modules",
        "Linking binaries"
    ],
    "test": [
        "Running unit tests",
        "Executing integration suite",
        "Collecting test results"
    ],
    "deploy": [
        "Uploading artifacts",
        "Restarting services",
        "Verifying health checks"
    ],
}

# Final message templates
FINAL_MSG = {
    "INFO": {
        "checkout": "Repository checked out successfully",
        "install": "All packages installed",
        "build": "Build completed in {duration_sec:.2f}s",
        "test": "All tests passed",
        "deploy": "Deployment succeeded"
    },
    "ERROR": {
        "checkout": "Checkout failed: network timeout",
        "install": "Install failed: dependency conflict",
        "build": "Build error: syntax error in module",
        "test": "Test suite failed: 1 test did not pass",
        "deploy": "Deployment error: insufficient permissions"
    }
}

def generate_run(run_id: int, start_ts: datetime, anomaly_prob: float):
    records = []
    ts = start_ts
    host = socket.gethostname()
    user = getpass.getuser()
    pid = os.getpid()
    thread = threading.get_ident()

    for stage in STAGES:
        # 1) Stage start
        records.append({
            "run_id": run_id,
            "timestamp": ts.isoformat(),
            "stage": stage,
            "status": "INFO",
            "message": f"Starting {stage}",
            "host": host,
            "user": user,
            "pid": pid,
            "thread": thread,
            "label": 0
        })

        ts += timedelta(seconds=random.uniform(1, 5))

        # 2) 0–2 intermediate logs
        for _ in range(random.randint(0, 2)):
            ts += timedelta(seconds=random.uniform(1, 10))
            msg = random.choice(INTERMEDIATE[stage])
            records.append({
                "run_id": run_id,
                "timestamp": ts.isoformat(),
                "stage": stage,
                "status": "INFO",
                "message": msg,
                "host": host,
                "user": user,
                "pid": pid,
                "thread": thread,
                "label": 0
            })

        # 3) Stage completion
        ts += timedelta(seconds=random.uniform(5, 30))
        is_anomaly = random.random() < anomaly_prob
        status = "ERROR" if is_anomaly else "INFO"
        # calculate duration of this stage
        duration = (ts - datetime.fromisoformat(records[-1]["timestamp"])).total_seconds()

        msg_template = FINAL_MSG[status][stage]
        msg = msg_template.format(duration_sec=duration)

        records.append({
            "run_id": run_id,
            "timestamp": ts.isoformat(),
            "stage": stage,
            "status": status,
            "message": msg,
            "host": host,
            "user": user,
            "pid": pid,
            "thread": thread,
            "duration_sec": duration,
            "label": int(is_anomaly)
        })

        # Slightly delay the start of the next stage
        ts += timedelta(seconds=random.uniform(1, 3))

    return records

def main():
    parser = argparse.ArgumentParser(description="Synthetic CI/CD log generator")
    parser.add_argument("-o", "--output-dir", default="logs", help="where to save files")
    parser.add_argument("-r", "--runs", type=int, default=100, help="number of runs")
    parser.add_argument("-p", "--anomaly-prob", type=float, default=0.05,
                        help="anomaly probability per stage")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_ts = datetime.now()

    for run_id in range(1, args.runs + 1):
        recs = generate_run(run_id, base_ts, args.anomaly_prob)
        fname = os.path.join(args.output_dir, f"run_{run_id:03d}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        # shift the start of the next run by 5-10 minutes
        base_ts += timedelta(minutes=random.uniform(5, 10))

    print(f"Generated {args.runs} files in '{args.output_dir}/'")

if __name__ == "__main__":
    main()