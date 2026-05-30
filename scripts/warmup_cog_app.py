#!/usr/bin/env python

import sys
import time
import argparse
import urllib.request
import urllib.error
import json


def make_request(url, data=None, method="GET", timeout=10):
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")

    encoded_data = None
    if data is not None:
        encoded_data = json.dumps(data).encode("utf-8")

    try:
        with urllib.request.urlopen(req, data=encoded_data, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8")
            return status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise e


def main():
    parser = argparse.ArgumentParser(description="Warm up Cog model container.")
    parser.add_argument("--model", required=True, help="Model type (e.g., wan22_t2v)")
    parser.add_argument("--port", type=int, default=5000, help="Host port of container")
    parser.add_argument("--timeout", type=int, default=300, help="Max wait timeout in seconds")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    health_url = f"{base_url}/health-check"
    predict_url = f"{base_url}/predictions"

    print(f"Polling {health_url} for readiness (timeout {args.timeout}s)...")
    start_time = time.time()
    ready = False

    while time.time() - start_time < args.timeout:
        try:
            status_code, body = make_request(health_url, timeout=5)
            if status_code == 200:
                data = json.loads(body)
                status = data.get("status")
                if status == "READY":
                    print("Container is READY.")
                    ready = True
                    break
                elif status == "SETUP_FAILED":
                    print("Error: Model setup failed inside the container.")
                    print(json.dumps(data, indent=2))
                    sys.exit(1)
                else:
                    print(f"Status is currently: {status}...")
            else:
                print(f"Health check returned status code: {status_code}")
        except Exception:
            # Server might not be bound to the port yet or is starting up
            print("Waiting for server to start...")

        time.sleep(5)

    if not ready:
        print("Error: Timeout waiting for container to become ready.")
        sys.exit(1)

    # Build the specific input payload for the model
    if args.model in ["qwen_image_edit"]:
        payload = {
            "input": {
                "prompt": "Put this logo on a big billboard",
                "images": ["https://buinity.com/cdn/platform/ai-model-providers/buinity-logo.webp"],
            }
        }
    elif args.model in ["wan22_i2v"]:
        payload = {
            "input": {
                "prompt": "Animate the logo",
                "image": "https://buinity.com/cdn/platform/ai-model-providers/buinity-logo.webp",
            }
        }
    else:
        payload = {"input": {"prompt": 'A big billboard with the text "Buinity"'}}

    print(f"Executing warmup prediction for model '{args.model}'...")
    try:
        status_code, body = make_request(predict_url, data=payload, method="POST", timeout=args.timeout)
        if status_code == 200:
            print("First-generation finished successfully. Cache populated.")
            print(f"Response (truncated): {body[:300]}...")
        else:
            print(f"Warning: Prediction run returned code {status_code}")
            print(f"Response: {body}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
