---
title: wandb login
---

**Usage**

`wandb login [OPTIONS] [KEY]...`

**Summary**

Verify and store your API key for authentication with W&B services.

By default, only store credentials locally without verifying them with W&B.
To verify credentials, set `--verify=True`.

For server deployments (dedicated cloud or customer-managed instances),
specify the host URL using the `--host` flag. You can also set environment
variables `WANDB_BASE_URL` and `WANDB_API_KEY` instead of running the
`login` command with host parameters.


**Options**

| **Option** | **Description** |
| :--- | :--- |
| `--cloud` | Login to the cloud instead of local |
| `--host, --base-url` | Login to a specific instance of W&B |
| `--relogin` | Force relogin if already logged in. |
| `--anonymously` | Log in anonymously |
| `--verify / --no-verify` | Verify login credentials |



