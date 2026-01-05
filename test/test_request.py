import requests
import os
import pytest

# Flask server URL
FLASK_APP_URL = "http://127.0.0.1:5002/treedata"

# Path to the test MSA file
MSA_FILE_PATH = "test-data/test_msa.fasta"

# Parameters for the analysis
WINDOW_SIZE = 100
STEP_SIZE = 50


def is_server_running(url: str, timeout: float = 2.0) -> bool:
    """Check if the Flask server is running by attempting a quick connection."""
    try:
        # Try a simple GET request with a short timeout
        base_url = url.rsplit("/", 1)[0]  # Get base URL without endpoint
        requests.get(base_url, timeout=timeout)
        return True
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


@pytest.mark.skipif(
    not is_server_running(FLASK_APP_URL),
    reason="Flask server is not running at " + FLASK_APP_URL,
)
def test_treedata_endpoint():
    """Test the /treedata endpoint with an MSA file upload."""
    if not os.path.exists(MSA_FILE_PATH):
        pytest.skip(f"Test MSA file not found: {MSA_FILE_PATH}")

    with open(MSA_FILE_PATH, "rb") as f:
        files = {
            "msaFile": (os.path.basename(MSA_FILE_PATH), f, "application/octet-stream")
        }

        # Prepare the form data
        data = {
            "windowSize": str(WINDOW_SIZE),
            "windowStepSize": str(STEP_SIZE),
            "midpointRooting": "on",
        }

        response = requests.post(FLASK_APP_URL, files=files, data=data, timeout=60)

        assert response.status_code == 200, (
            f"Expected 200 OK, got {response.status_code}"
        )

        payload = response.json()
        assert isinstance(payload, dict), "Response should be a JSON object"
        print(f"Payload keys: {list(payload.keys())}")
