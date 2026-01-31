import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from webapp import create_app
from io import BytesIO


def test_sse_streaming():
    print("Initializing app...")
    app = create_app()
    client = app.test_client()

    # 1. Load test data
    tree_path = "datasets/small_example copy 3.tree"
    if not os.path.exists(tree_path):
        print(f"Error: {tree_path} not found")
        return

    with open(tree_path, "rb") as f:
        tree_content = f.read()

    # 2. Start processing via /treedata/stream
    data = {
        "treeFile": (BytesIO(tree_content), "test.tree"),
        "enableRooting": "false",
        "windowSize": "1",
        "windowStep": "1",
    }

    print("POST /treedata/stream...")
    response = client.post(
        "/treedata/stream", data=data, content_type="multipart/form-data"
    )

    if response.status_code != 200:
        print(f"Error starting stream: {response.status_code}")
        print(response.get_data(as_text=True))
        return

    result = response.get_json()
    channel_id = result["channel_id"]
    print(f"Got channel_id: {channel_id}")

    # 3. Listen to SSE stream
    print(f"GET /stream/progress/{channel_id}...")
    response = client.get(f"/stream/progress/{channel_id}")

    if response.status_code != 200:
        print(f"Error connecting to stream: {response.status_code}")
        return

    # Read the stream
    messages = []
    print("Listening for events...")

    # Flask test client response is an iterable of bytes
    for chunk in response.response:
        chunk_str = chunk.decode("utf-8")
        # SSE messages might be split across chunks or multiple in one chunk
        lines = chunk_str.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    msg = json.loads(data_str)
                    messages.append(msg)

                    if "percent" in msg:
                        print(f"Progress: {msg['percent']}% - {msg.get('message', '')}")
                    elif "data" in msg:
                        print("Received COMPLETE event with data")
                        # Verify data structure
                        if "interpolated_trees" in msg["data"]:
                            print(
                                f"  - Interpolated trees: {len(msg['data']['interpolated_trees'])}"
                            )
                        break
                    elif "error" in msg:
                        print(f"Received ERROR: {msg['error']}")
                        break

                except json.JSONDecodeError:
                    pass
            elif line.startswith("event:"):
                event_type = line[6:].strip()
                if event_type == "complete":
                    print("Event: complete")
                elif event_type == "error":
                    print("Event: error")

    # 4. Verify progress
    progress_values = [m.get("percent") for m in messages if "percent" in m]
    print(f"\nProgress sequence: {progress_values}")

    if not progress_values:
        print("FAILED: No progress updates received")
        return

    if progress_values[-1] != 100:
        print("FAILED: Did not reach 100%")
        return

    if len(progress_values) < 3:
        print("WARNING: Very few progress updates received")

    print("\nTest PASSED!")


if __name__ == "__main__":
    test_sse_streaming()
