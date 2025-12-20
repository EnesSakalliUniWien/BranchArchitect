import requests
import os
import json
import time

# Flask server URL
FLASK_APP_URL = "http://127.0.0.1:5002/treedata"

# Path to the test MSA file
MSA_FILE_PATH = "test-data/test_msa.fasta"

# Parameters for the analysis
WINDOW_SIZE = 100
STEP_SIZE = 50

# Prepare the files dictionary for requests
with open(MSA_FILE_PATH, "rb") as f:
    files = {"msaFile": (os.path.basename(MSA_FILE_PATH), f, "application/octet-stream")}
    
    # Prepare the form data
    data = {
        "windowSize": str(WINDOW_SIZE),
        "windowStepSize": str(STEP_SIZE),
        "midpointRooting": "on" # Example parameter
    }

    print(f"Sending POST request to {FLASK_APP_URL}")
    print(f"Uploading file: {MSA_FILE_PATH}")
    print(f"With data: {data}")

    try:
        response = requests.post(FLASK_APP_URL, files=files, data=data)
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Body: {response.text}") # Print raw text instead of JSON

        # Expected: 200 OK with full interpolation payload
        if response.status_code == 200:
            try:
                payload = response.json()
                print("Successfully received 200 OK.")
                print(f"Payload keys: {list(payload.keys())}")
            except json.JSONDecodeError:
                print("Received 200 OK but failed to parse JSON response.")
        else:
            print("Unexpected response status code.")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the Flask server. Is it running?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
