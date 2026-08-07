'''
Retrieve metrics about your calls
'''

import requests
import json
import os

# Weave API URL
url = "https://trace.wandb.ai/calls/stats"

# Configure the types of metrics to retrieve for a specified time range
payload = {
    "project_id": "<your-team-name/your-project-name>",
# Specify time range
    "start": "2026-03-01T00:00:00Z",
    "end": "2026-03-10T00:00:00Z",
# Specify the size of the buckets, in seconds.
    "granularity": 86400,
    "filter": {
        "trace_roots_only": True,
        "op_names": ["web_app"]
    },
# Specify metrics and their aggregate function
    "usage_metrics": [
        {"metric": "total_tokens", "aggregations": ["sum"]},
        {"metric": "total_cost", "aggregations": ["sum"]}
    ],
    "call_metrics": [
        {"metric": "call_count", "aggregations": ["sum"]},
        {"metric": "error_count", "aggregations": ["sum"]},
        {"metric": "latency_ms", "aggregations": ["avg", "min", "max"], "percentiles": [50, 95, 99]}
    ]
}

API_KEY = os.getenv("WANDB_API_KEY")

response = requests.post(url, json=payload, auth=("api", API_KEY))

print(json.dumps(response.json(), indent=2))