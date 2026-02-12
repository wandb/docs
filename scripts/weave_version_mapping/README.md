# Mapping W&B Server Releases to Weave Versions

This directory contains a script to help determine which Weave version (or version range) is included in a given W&B Server (Core) release.

## Overview

W&B Server (`wandb/core` repository) uses Weave functionality through a git submodule located at `services/weave-python/weave-public`. This submodule points to a specific commit in the [Weave repository](https://github.com/wandb/weave).

To determine which Weave version is included in a Core release:

1. **Check the git submodule commit** in the Core release branch (via GitHub API)
2. **Map that commit to a Weave version** by:
   - Reading the version from `weave/version.py` in that commit (via GitHub API)
   - Finding the nearest Weave release tags
   - Determining how many commits ahead/behind the commit is from release tags

**No local repository clones required!** The script uses the GitHub API, so you only need:
- GitHub CLI (`gh`) installed and authenticated, OR
- `GITHUB_TOKEN` environment variable set

## Python Script

```bash
python scripts/map_core_to_weave_version.py <core-version>
```

**Prerequisites:**
- Python 3 with `requests` library: `pip install requests`
- GitHub authentication (choose one):
  - GitHub CLI (`gh`) installed and authenticated: `gh auth login`
  - OR set `GITHUB_TOKEN` environment variable

**Examples:**
```bash
# Check a specific version
python scripts/map_core_to_weave_version.py v0.77.0
python scripts/map_core_to_weave_version.py v0.77.1

# Use release branch name directly
python scripts/map_core_to_weave_version.py server-release-0.77.x

# Check in-development Server versions
python scripts/map_core_to_weave_version.py master
python scripts/map_core_to_weave_version.py server-release-0.78.x

# Short output (only version number, for scripting)
python scripts/map_core_to_weave_version.py v0.77.0 --short
WEAVE_VERSION=$(python scripts/map_core_to_weave_version.py v0.77.0 --short)

# List all available release branches
python scripts/map_core_to_weave_version.py --list

# Verbose output with more details
python scripts/map_core_to_weave_version.py --verbose
```

## How It Works

1. **Version Normalization**: Converts version strings like `v0.77.0` to release branch names like `server-release-0.77.x`. You can also pass branch names directly (e.g., `master` or `server-release-0.78.x`) to check in-development versions.

2. **Submodule Lookup**: Uses `git ls-tree` to find the commit hash that the `weave-public` submodule points to in the specified release branch

3. **Version Extraction**: 
   - Reads `weave/version.py` from that commit to get the version string
   - **Important**: All versions in `version.py` have a `-dev0` suffix, but this maps to the PyPI version without the suffix (e.g., `0.52.23-dev0` in version.py = `0.52.23` on PyPI)
   - Uses `git describe` to find the nearest release tag
   - Lists all Weave release tags that contain this commit

4. **Version Range**: Determines if the commit is:
   - Exactly on a release tag
   - A certain number of commits after a release tag
   - Between two release tags

## Output Modes

### Short Output (`--short` or `-s`)

For scripting and automation, use the `--short` flag to output only the Weave version number:

```bash
$ python scripts/map_core_to_weave_version.py v0.77.0 --short
0.52.23
```

This makes it easy to use in scripts:
```bash
WEAVE_VERSION=$(python scripts/map_core_to_weave_version.py v0.77.0 --short)
echo "Using Weave version: $WEAVE_VERSION"
```

### Default Output

Default output:
```text
✓ Found Weave submodule commit: 4c83976027041e61e7689e816c5a1fafca7d0ee8
Info: Fetching Weave version information (this may take 10-20 seconds)...
Info: Fetching Weave version from commit: 4c83976027041e61e7689e816c5a1fafca7d0ee8
==========================================
Summary:
  Core v0.77.0 includes Weave version: 0.52.23
  (Based on Weave commit at: v0.52.26)
==========================================
```

Verbose output:
```text
==========================================
Core Release: v0.77.0
Release Branch: server-release-0.77.x
==========================================

Weave Submodule Commit: 4c83976027041e61e7689e816c5a1fafca7d0ee8
Weave Version (from version.py): 0.52.23-dev0
Weave Version (PyPI equivalent): 0.52.23
Git Describe: v0.52.22-12-g4c83976027
Commit Info: 4c83976027 chore(weave): Add ty type checker (#5853)

Weave Release Tags containing this commit:
  - v0.52.23
  - v0.52.24
  - v0.52.25
  - v0.52.26

Latest Weave release tag: v0.52.26

==========================================
Summary:
  Core v0.77.0 includes Weave version: 0.52.23
  (version.py shows: 0.52.23-dev0, PyPI release: 0.52.23)
  (Based on Weave commit at: v0.52.22 + 12 commit(s))
==========================================
```

## Understanding the Results

- **Weave Version (from version.py)**: The version string from `weave/version.py`. All versions in the Weave repo include a `-dev0` suffix.

- **Weave Version (PyPI equivalent)**: The corresponding version on PyPI, which is the same version without the `-dev0` suffix. This is the actual published version that Core is using.

- **Git Describe**: Shows the commit relative to the nearest tag (e.g., `v0.52.22-12-g4c83976027` means 12 commits after v0.52.22).

- **Release Tags**: All Weave release tags that contain this commit. If a commit is included in multiple tags, it means those tags all point to commits that include this one.

- **Version Range**: The most accurate description of where this commit falls in the Weave release timeline.

## Important Notes

1. **Submodule Commits**: The submodule commit in a Core release branch points to a specific commit in the Weave repository. This commit corresponds to a published Weave version on PyPI.

2. **Version Format**: All Weave versions in `version.py` have a `-dev0` suffix (this is how versions are tracked in the git repository). However, this maps directly to the PyPI version without the suffix. For example:
   - `0.52.23-dev0` in `version.py` = `0.52.23` on PyPI
   - This means Core is using published Weave versions, not development/unpublished versions

3. **Multiple Tags**: If a commit appears in multiple release tags, it means those tags are all descendants of this commit. The "latest" tag is the most recent one.

4. **Patch Versions**: Core patch releases (e.g., v0.77.1) may update the Weave submodule to a newer commit, so you should check each patch version separately.

5. **In-Development Versions**: You can check the Weave version in development branches by passing the branch name directly:
   - Early in development: `python scripts/map_core_to_weave_version.py master`
   - After release branch is cut: `python scripts/map_core_to_weave_version.py server-release-0.78.x`
   - The script will show the branch name in the summary instead of a version number

## Authentication

The script requires GitHub authentication to query private repositories. You can provide it in two ways:

1. **GitHub CLI (recommended)**: Install and authenticate:
   ```bash
   gh auth login
   ```

2. **Environment variable**: Set a GitHub personal access token:
   ```bash
   export GITHUB_TOKEN=your_token_here
   python scripts/map_core_to_weave_version.py v0.77.0
   ```

The script will automatically detect and use either method.

## How It Works

The script uses the GitHub REST API to:

1. **Get the submodule commit**: Fetches the tree for the Core release branch and finds the `weave-public` submodule entry
2. **Get Weave version**: Fetches `weave/version.py` from that commit using the GitHub Contents API
3. **Find tags**: Lists Weave repository tags and determines which contain the commit using the Compare API
4. **Calculate version range**: Determines the nearest release tag and commit offset

All operations happen via API calls - no local repository clones needed!

## Troubleshooting

- **"GitHub authentication required"**: 
  - Run `gh auth login` to authenticate with GitHub CLI, OR
  - Set `GITHUB_TOKEN` environment variable with a valid token

- **"Release branch not found"**: The branch name might be incorrect. Use `--list` to see available branches.

- **"Could not find weave-public submodule"**: The submodule path may have changed in newer Core versions.

- **"Could not extract version"**: The Weave repo structure may have changed. The commit might not exist or the file path changed.

- **Rate limiting**: GitHub API has rate limits. If you hit limits, wait a bit or use a token with higher rate limits.

- **Network issues**: The script requires internet access to query the GitHub API.
