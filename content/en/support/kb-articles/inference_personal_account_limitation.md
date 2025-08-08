---
url: /support/:filename
title: "Why isn't W&B Inference available for my personal account?"
toc_hide: true
type: docs
support:
  - inference
---

Personal accounts don't support W&B Inference. You'll see this 429 error: "W&B Inference isn't available for personal accounts. Please switch to a non-personal account to access W&B Inference."

## Background

Personal entities were deprecated in May 2024. This only affects legacy accounts that still use personal entities.

## How to access W&B Inference

### Create a Team

1. Log in to your W&B account
2. Click on your profile icon in the top right
3. Select "Create new team"
4. Choose a team name
5. Use this team for W&B Inference requests

### Update your code

Change from personal entity to team:

**Before (doesn't work):**
```python
project="your-username/project-name"  # Personal entity
```

**After (works):**
```python
project="your-team/project-name"  # Team entity
```

## Benefits of using Teams

- Access to W&B Inference
- Better collaboration features
- Shared projects and resources
- Team-based billing and usage tracking

## Need help?

If you're having trouble creating a team or switching from a personal account, contact W&B support for assistance. 