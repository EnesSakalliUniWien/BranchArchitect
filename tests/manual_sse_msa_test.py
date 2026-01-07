import time
import json
import sys
import os
import io
from pathlib import Path
import logging

# Suppress verbose logs
logging.getLogger("brancharchitect").setLevel(logging.WARNING)
logging.getLogger("webapp").setLevel(logging.WARNING)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from webapp import create_app


def listen_to_stream(client, channel_id):
    print(f"Listening to stream {channel_id}...")
    response = client.get(f"/stream/progress/{channel_id}")

    if response.status_code != 200:
        print(f"Error connecting to stream: {response.status_code}")
        print(response.data)
        return False

    progress_events = []
    complete = False
    buffer = ""

    for chunk in response.response:
        chunk_str = chunk.decode("utf-8")
        buffer += chunk_str

        while "\n\n" in buffer:
            message, buffer = buffer.split("\n\n", 1)

            # Parse message lines
            event_type = None
            data_str = ""

            for line in message.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_str += line.split(":", 1)[1].strip()

            if not data_str:
                continue

            try:
                data = json.loads(data_str)

                if "percent" in data:
                    pct = data["percent"]
                    msg = data.get("message", "")
                    print(f"Progress: {pct}% - {msg}")
                    progress_events.append(pct)

                if "error" in data:
                    print(f"Error event: {data['error']}")
                    return False

                if event_type == "complete" or "data" in data:  # Complete event
                    print("Received COMPLETE event")
                    if "data" in data and isinstance(data["data"], dict):
                        print(f"Data keys: {list(data['data'].keys())}")
                    complete = True

            except json.JSONDecodeError:
                print(f"Failed to decode JSON: {data_str[:100]}...")

    if complete:
        print(f"Stream completed successfully. Progress sequence: {progress_events}")
        if 100 in progress_events:
            return True
        else:
            print("FAILED: Did not reach 100%")
            return False
    else:
        print("Stream ended without complete event")
        return False


def test_msa_only(client):
    print("\n=== TEST 1: MSA Only ===")
    msa_path = Path("datasets/hiv_rt.fasta")
    if not msa_path.exists():
        print(f"Skipping MSA test: {msa_path} not found")
        return

    with open(msa_path, "rb") as f:
        msa_content = f.read()

    data = {
        "msaFile": (io.BytesIO(msa_content), "hiv_rt.fasta"),
        "windowSize": 200,  # Larger window for faster processing
        "windowStepSize": 100,
        "midpointRooting": "on",
    }

    print("POST /treedata/stream (MSA Only)...")
    res = client.post("/treedata/stream", data=data, content_type="multipart/form-data")

    if res.status_code != 200:
        print(f"Request failed: {res.status_code}")
        print(res.data)
        return

    channel_id = res.json["channel_id"]
    print(f"Got channel_id: {channel_id}")

    if listen_to_stream(client, channel_id):
        print("TEST 1 PASSED")
    else:
        print("TEST 1 FAILED")


def test_tree_and_msa(client):
    print("\n=== TEST 2: Tree + MSA ===")
    tree_path = Path("datasets/hiv_rt.newick")
    msa_path = Path("datasets/hiv_rt.fasta")

    if not tree_path.exists() or not msa_path.exists():
        print("Skipping Tree+MSA test: files not found")
        return

    with open(tree_path, "rb") as f:
        tree_content = f.read()
    with open(msa_path, "rb") as f:
        msa_content = f.read()

    data = {
        "treeFile": (io.BytesIO(tree_content), "hiv_rt.newick"),
        "msaFile": (io.BytesIO(msa_content), "hiv_rt.fasta"),
        "windowSize": 1,
        "windowStepSize": 1,
        "midpointRooting": "on",
    }

    print("POST /treedata/stream (Tree + MSA)...")
    res = client.post("/treedata/stream", data=data, content_type="multipart/form-data")

    if res.status_code != 200:
        print(f"Request failed: {res.status_code}")
        print(res.data)
        return

    channel_id = res.json["channel_id"]
    print(f"Got channel_id: {channel_id}")

    if listen_to_stream(client, channel_id):
        print("TEST 2 PASSED")
    else:
        print("TEST 2 FAILED")


if __name__ == "__main__":
    print("Initializing app...")
    app = create_app()
    client = app.test_client()

    test_msa_only(client)
    test_tree_and_msa(client)
