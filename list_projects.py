#!/usr/bin/env python3
"""List projects in your W&B account."""

try:
    from wandb.apis.public.api import Api
    
    # Initialize the API
    api = Api()
    
    # Get projects for the default entity
    print("Fetching projects from your W&B account...\n")
    
    projects = api.projects()
    
    project_list = list(projects)
    
    if not project_list:
        print("No projects found in your account.")
    else:
        print(f"Found {len(project_list)} project(s):\n")
        for i, project in enumerate(project_list, 1):
            print(f"{i}. {project.name}")
            print(f"   Entity: {project.entity}")
            print(f"   URL: {project.url}")
            if hasattr(project, 'created_at') and project.created_at:
                print(f"   Created: {project.created_at}")
            print()
            
except ImportError:
    print("Error: wandb package is not installed.")
    print("Install it with: pip install wandb")
except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure you're logged in to W&B:")
    print("  wandb login")
    print("\nOr set your API key:")
    print("  export WANDB_API_KEY=your_api_key")
