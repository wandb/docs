#!/usr/bin/env python3
"""
Script to map W&B Server (Core) releases to Weave versions.

This script checks the git submodule commit in Core release branches
and maps it to the corresponding Weave version using the GitHub API.

Usage:
    python map_core_to_weave_version.py <core-version>
    python map_core_to_weave_version.py v0.77.0
    python map_core_to_weave_version.py v0.77.1
    python map_core_to_weave_version.py server-release-0.77.x

Or list all release branches:
    python map_core_to_weave_version.py --list

Requirements:
    - GitHub CLI (gh) installed and authenticated, OR
    - GITHUB_TOKEN environment variable set for API access
"""

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import json

# Suppress urllib3 warnings about OpenSSL
# Redirect stderr during import to suppress the warning
import io
import contextlib

stderr_buffer = io.StringIO()
with contextlib.redirect_stderr(stderr_buffer):
    try:
        import requests
        import urllib3
        urllib3.disable_warnings()
    except ImportError:
        print("Error: 'requests' library is required. Install with: pip install requests", file=sys.stderr)
        sys.exit(1)

# Also set up warnings filter
warnings.filterwarnings("ignore", message=".*urllib3.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*OpenSSL.*", category=UserWarning)


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


CORE_REPO = "wandb/core"
WEAVE_REPO = "wandb/weave"


def error(msg: str) -> None:
    print(f"{Colors.RED}Error:{Colors.NC} {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"{Colors.BLUE}Info:{Colors.NC} {msg}")


def success(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.NC} {msg}")


def warning(msg: str) -> None:
    print(f"{Colors.YELLOW}Warning:{Colors.NC} {msg}")


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment or gh CLI."""
    # Check environment variable first
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    
    # Try to get from gh CLI
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def github_api_request(url: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Make a GitHub API request."""
    if not token:
        token = get_github_token()
    
    if not token:
        error("GitHub authentication required. Set GITHUB_TOKEN or run 'gh auth login'")
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_core_release_branch(version: str) -> str:
    """Convert version string to release branch name."""
    # Handle v0.XX.Y format
    match = re.match(r'^v?0\.(\d+)\.(\d+)$', version)
    if match:
        major = match.group(1)
        return f"server-release-0.{major}.x"
    
    # Handle server-release-0.XX.x format
    if version.startswith("server-release-"):
        return version
    
    error(f"Invalid Core version format: {version}\n"
          f"Expected format: v0.XX.Y or server-release-0.XX.x")


def get_submodule_commit(release_branch: str, verbose: bool = False) -> str:
    """Get the weave-public submodule commit for a given branch using GitHub API."""
    if verbose:
        info(f"Fetching submodule info from GitHub for branch: {release_branch}")
    
    # Get the tree for the release branch
    # First, get the branch SHA
    branch_url = f"https://api.github.com/repos/{CORE_REPO}/branches/{release_branch}"
    try:
        branch_data = github_api_request(branch_url)
        commit_sha = branch_data["commit"]["sha"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error(f"Release branch {release_branch} not found")
        raise
    
    # Get the tree with recursive option to include submodules
    tree_url = f"https://api.github.com/repos/{CORE_REPO}/git/trees/{commit_sha}?recursive=1"
    tree_data = github_api_request(tree_url)
    
    # Find the weave-public submodule entry
    submodule_path = "services/weave-python/weave-public"
    for item in tree_data.get("tree", []):
        if item.get("path") == submodule_path and item.get("type") == "commit":
            submodule_commit = item.get("sha")
            if submodule_commit:
                success(f"Found Weave submodule commit: {submodule_commit}")
                return submodule_commit
    
    error(f"Could not find weave-public submodule in {release_branch}")


def get_weave_version(commit: str) -> Tuple[str, str]:
    """
    Get Weave version from version.py file using GitHub API.
    Returns: (version_from_file, pypi_version)
    """
    info(f"Fetching Weave version from commit: {commit}")
    
    # Get the file content from GitHub API
    file_url = f"https://api.github.com/repos/{WEAVE_REPO}/contents/weave/version.py?ref={commit}"
    try:
        file_data = github_api_request(file_url)
        
        # Decode base64 content
        import base64
        content = base64.b64decode(file_data["content"]).decode("utf-8")
        
        # Extract VERSION = "X.Y.Z" or "X.Y.Z-dev0"
        match = re.search(r'VERSION\s*=\s*"([^"]+)"', content)
        if match:
            version_from_file = match.group(1)
            # Remove -dev0 suffix to get PyPI version
            pypi_version = version_from_file.replace("-dev0", "")
            return (version_from_file, pypi_version)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            warning(f"Could not find version.py in commit {commit}")
        else:
            warning(f"Error fetching version.py: {e}")
    
    return ("unknown", "unknown")


def get_git_describe(commit: str) -> str:
    """Get git describe info using GitHub API to find nearest tag.
    
    Optimized to only check recent tags (most recent 30).
    """
    # Get recent tags (most recent 30 should be enough to find the nearest)
    tags_url = f"https://api.github.com/repos/{WEAVE_REPO}/tags?per_page=30"
    try:
        tags_data = github_api_request(tags_url)
        
        # Filter and sort tags by version
        version_tags = []
        for tag in tags_data:
            tag_name = tag["name"]
            if re.match(r'^v\d+\.\d+\.\d+$', tag_name):
                version_tags.append(tag_name)
        
        # Sort by version number (oldest first for finding nearest)
        version_tags.sort(key=lambda x: tuple(map(int, x[1:].split('.'))))
        
        # Find the most recent tag that the commit is after
        nearest_tag = None
        for tag in reversed(version_tags):
            # Get the tag commit - handle both annotated and direct tags
            tag_url = f"https://api.github.com/repos/{WEAVE_REPO}/git/refs/tags/{tag}"
            try:
                tag_ref = github_api_request(tag_url)
                tag_obj = tag_ref["object"]
                if tag_obj["type"] == "tag":
                    tag_commit_url = f"https://api.github.com/repos/{WEAVE_REPO}/git/tags/{tag_obj['sha']}"
                    tag_commit_data = github_api_request(tag_commit_url)
                    tag_sha = tag_commit_data["object"]["sha"]
                else:
                    tag_sha = tag_obj["sha"]
                
                # Check if commit is after this tag
                compare_url = f"https://api.github.com/repos/{WEAVE_REPO}/compare/{tag_sha}...{commit}"
                try:
                    compare_data = github_api_request(compare_url)
                    status = compare_data.get("status")
                    if status == "ahead":
                        ahead_by = compare_data.get("ahead_by", 0)
                        nearest_tag = (tag, ahead_by)
                        break
                    elif status == "identical":
                        return tag
                except requests.exceptions.HTTPError:
                    continue
            except requests.exceptions.HTTPError:
                continue
        
        if nearest_tag:
            tag, ahead_by = nearest_tag
            return f"{tag}-{ahead_by}-g{commit[:12]}"
        
        # Fallback: just return the commit short hash
        return f"unknown-{commit[:12]}"
    except Exception as e:
        warning(f"Could not determine git describe: {e}")
        return "unknown"


def get_commit_info(commit: str) -> str:
    """Get one-line commit info using GitHub API."""
    commit_url = f"https://api.github.com/repos/{WEAVE_REPO}/commits/{commit}"
    try:
        commit_data = github_api_request(commit_url)
        message = commit_data["commit"]["message"].split("\n")[0]
        return f"{commit[:12]} {message}"
    except Exception:
        return "unknown"


def get_tags_containing(commit: str) -> List[str]:
    """Get all release tags containing this commit using GitHub API.
    
    Optimized to only check recent tags (most recent 30) since tags contain
    all commits that came before them, so we only need to check tags created
    after the commit.
    """
    # Get recent tags (most recent 30 should be enough)
    tags_url = f"https://api.github.com/repos/{WEAVE_REPO}/tags?per_page=30"
    try:
        tags_data = github_api_request(tags_url)
        
        containing_tags = []
        for tag in tags_data:
            tag_name = tag["name"]
            if not re.match(r'^v\d+\.\d+\.\d+$', tag_name):
                continue
            
            # Get the tag commit - tags can be direct commits or annotated tags
            tag_url = f"https://api.github.com/repos/{WEAVE_REPO}/git/refs/tags/{tag_name}"
            try:
                tag_ref = github_api_request(tag_url)
                # Handle both annotated tags and direct commits
                tag_obj = tag_ref["object"]
                if tag_obj["type"] == "tag":
                    # Annotated tag - get the commit it points to
                    tag_commit_url = f"https://api.github.com/repos/{WEAVE_REPO}/git/tags/{tag_obj['sha']}"
                    tag_commit_data = github_api_request(tag_commit_url)
                    tag_sha = tag_commit_data["object"]["sha"]
                else:
                    # Direct commit reference
                    tag_sha = tag_obj["sha"]
                
                # Check if commit is reachable from tag
                # Compare commit to tag: if tag is ahead or identical, tag contains commit
                compare_url = f"https://api.github.com/repos/{WEAVE_REPO}/compare/{commit}...{tag_sha}"
                try:
                    compare_data = github_api_request(compare_url)
                    status = compare_data.get("status")
                    # "ahead" means tag is ahead of commit (tag contains commit)
                    # "identical" means they're the same
                    if status in ["ahead", "identical"]:
                        containing_tags.append(tag_name)
                except requests.exceptions.HTTPError as e:
                    # 404 means commits are not related
                    if e.response.status_code != 404:
                        continue
            except requests.exceptions.HTTPError:
                continue
        
        return sorted(containing_tags, key=lambda x: tuple(map(int, x[1:].split('.'))))
    except Exception:
        return []


def get_version_range(commit: str, tags: List[str]) -> str:
    """Determine the version range for a commit.
    
    Optimized to only check recent tags (most recent 30).
    """
    if not tags:
        return "unknown"
    
    latest_tag = tags[-1]
    
    # Find the tag just before this commit
    # Get recent tags (most recent 30 should be enough)
    tags_url = f"https://api.github.com/repos/{WEAVE_REPO}/tags?per_page=30"
    try:
        tags_data = github_api_request(tags_url)
        all_tags = []
        for tag in tags_data:
            tag_name = tag["name"]
            if re.match(r'^v\d+\.\d+\.\d+$', tag_name):
                all_tags.append(tag_name)
        
        all_tags.sort(key=lambda x: tuple(map(int, x[1:].split('.'))))
        
        # Find the tag just before our commit
        prev_tag = None
        for tag in reversed(all_tags):
            tag_url = f"https://api.github.com/repos/{WEAVE_REPO}/git/refs/tags/{tag}"
            try:
                tag_ref = github_api_request(tag_url)
                tag_sha = tag_ref["object"]["sha"]
                
                compare_url = f"https://api.github.com/repos/{WEAVE_REPO}/compare/{tag_sha}...{commit}"
                compare_data = github_api_request(compare_url)
                if compare_data.get("status") == "ahead":
                    prev_tag = tag
                    ahead_by = compare_data.get("ahead_by", 0)
                    if ahead_by > 0:
                        return f"{tag} + {ahead_by} commit(s)"
                    return tag
            except requests.exceptions.HTTPError:
                continue
        
        return latest_tag
    except Exception:
        return latest_tag


def list_release_branches() -> None:
    """List all available release branches using GitHub API."""
    branches_url = f"https://api.github.com/repos/{CORE_REPO}/branches?per_page=100"
    try:
        branches_data = github_api_request(branches_url)
        
        release_branches = []
        for branch in branches_data:
            branch_name = branch["name"]
            if "server-release" in branch_name or "release" in branch_name:
                release_branches.append(branch_name)
        
        print("Available Core release branches:")
        for branch in sorted(release_branches):
            print(f"  {branch}")
    except Exception as e:
        error(f"Could not list branches: {e}")


def main():
    # Parse arguments
    verbose = False
    args = []
    for arg in sys.argv[1:]:
        if arg in ["--verbose", "-v"]:
            verbose = True
        elif arg in ["--help", "-h"]:
            print("Usage: python map_core_to_weave_version.py <core-version> [--verbose]")
            print("\nExamples:")
            print("  python map_core_to_weave_version.py v0.77.0")
            print("  python map_core_to_weave_version.py v0.77.1")
            print("  python map_core_to_weave_version.py server-release-0.77.x")
            print("  python map_core_to_weave_version.py v0.77.0 --verbose")
            print("\nOptions:")
            print("  --verbose, -v    Show detailed information")
            print("  --list, -l       List all available release branches")
            print("  --help, -h       Show this help message")
            print("\nNote: Requires GitHub CLI (gh) or GITHUB_TOKEN environment variable")
            sys.exit(0)
        else:
            args.append(arg)
    
    if len(args) < 1:
        error("Usage: python map_core_to_weave_version.py <core-version> [--verbose] or --list")
        print("\nUse --help for more information")
        sys.exit(1)
    
    if args[0] in ["--list", "-l"]:
        list_release_branches()
        return
    
    core_version = args[0]
    release_branch = get_core_release_branch(core_version)
    
    if verbose:
        info(f"Core version {core_version} maps to release branch: {release_branch}")
    
    # Check authentication
    token = get_github_token()
    if not token:
        error("GitHub authentication required. Set GITHUB_TOKEN or run 'gh auth login'")
    
    # Get submodule commit
    submodule_commit = get_submodule_commit(release_branch, verbose=verbose)
    
    # Get Weave version info
    if verbose:
        info(f"Fetching Weave version from commit: {submodule_commit}")
    else:
        info("Fetching Weave version information (this may take 10-20 seconds)...")
    
    weave_version_file, weave_version_pypi = get_weave_version(submodule_commit)
    git_describe = get_git_describe(submodule_commit)
    commit_info = get_commit_info(submodule_commit)
    tags = get_tags_containing(submodule_commit)
    version_range = get_version_range(submodule_commit, tags)
    
    # Print results
    if verbose:
        print("\n" + "=" * 42)
        print(f"Core Release: {core_version}")
        print(f"Release Branch: {release_branch}")
        print("=" * 42)
        print()
        print(f"Weave Submodule Commit: {submodule_commit}")
        print(f"Weave Version (from version.py): {weave_version_file}")
        if weave_version_file != weave_version_pypi:
            print(f"Weave Version (PyPI equivalent): {weave_version_pypi}")
        print(f"Git Describe: {git_describe}")
        print(f"Commit Info: {commit_info}")
        print()
        
        if tags:
            print("Weave Release Tags containing this commit:")
            print("  (Note: Only the earliest tag in this list represents the Weave version")
            print("   included in this Core release. Later tags also contain this commit")
            print("   because they are descendants of it.)")
            for tag in tags:
                print(f"  - {tag}")
            print()
            
            latest_tag = tags[-1]
            print(f"Latest Weave release tag: {latest_tag}")
            
            # Check if exactly on tag
            tag_url = f"https://api.github.com/repos/{WEAVE_REPO}/git/refs/tags/{latest_tag}"
            try:
                tag_ref = github_api_request(tag_url)
                tag_sha = tag_ref["object"]["sha"]
                if tag_sha == submodule_commit:
                    success(f"Commit is exactly on Weave release tag: {latest_tag}")
            except Exception:
                pass
        else:
            warning("No Weave release tags found containing this commit")
        print()
    
    # Always show summary
    print("=" * 42)
    print("Summary:")
    print(f"  Core {core_version} includes Weave version: {weave_version_pypi}")
    if version_range != "unknown":
        print(f"  (Based on Weave commit at: {version_range})")
    print("=" * 42)
    
    if verbose:
        print(f"\nDetailed info:")
        print(f"  - version.py shows: {weave_version_file}")
        print(f"  - Git describe: {git_describe}")
        print(f"  - Commit: {commit_info}")
        if tags:
            print(f"  - Earliest containing tag: {tags[0]}")
            if len(tags) > 1:
                print(f"  - Latest containing tag: {tags[-1]}")


if __name__ == "__main__":
    main()
