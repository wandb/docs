#!/usr/bin/env python3
"""List projects using W&B API directly (fallback if MCP server isn't connected)."""

import os
import json
import urllib.request
import urllib.error

api_key = os.environ.get('WANDB_API_KEY')
if not api_key:
    print("Error: WANDB_API_KEY environment variable is not set.")
    exit(1)

# Use W&B Public API to list projects
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

def make_request(url):
    """Make an HTTP GET request with headers."""
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

# Get the default entity (user) first
print("Fetching your W&B account information...\n")
try:
    user_data = make_request('https://api.wandb.ai/api/v1/user')
    entity = user_data.get('username') or user_data.get('entity', {}).get('username')
    
    if not entity:
        print("Could not determine entity. Trying to list projects...")
        entity = None
    else:
        print(f"Entity: {entity}\n")
    
    # List projects
    print("Fetching projects...\n")
    if entity:
        projects_url = f'https://api.wandb.ai/api/v1/projects?entity={entity}'
    else:
        projects_url = 'https://api.wandb.ai/api/v1/projects'
    
    projects_data = make_request(projects_url)
    projects = projects_data.get('projects', [])
    
    if not projects:
        print("No projects found in your account.")
    else:
        print(f"Found {len(projects)} project(s):\n")
        for i, project in enumerate(projects, 1):
            project_name = project.get('name', 'Unknown')
            project_entity = project.get('entity', {}).get('name', entity or 'Unknown')
            project_id = project.get('id', 'Unknown')
            
            print(f"{i}. {project_name}")
            print(f"   Entity: {project_entity}")
            print(f"   Project ID: {project_id}")
            if project.get('createdAt'):
                print(f"   Created: {project['createdAt']}")
            print()
            
except urllib.error.HTTPError as e:
    print(f"Error connecting to W&B API: HTTP {e.code}")
    if e.code == 401:
        print("Authentication failed. Check your WANDB_API_KEY.")
    else:
        print(f"Response: {e.read().decode()}")
except urllib.error.URLError as e:
    print(f"Error connecting to W&B API: {e}")
    print("\nMake sure:")
    print("1. Your WANDB_API_KEY is valid")
    print("2. You have internet connectivity")
    print("3. The W&B API is accessible")
except Exception as e:
    print(f"Unexpected error: {e}")
