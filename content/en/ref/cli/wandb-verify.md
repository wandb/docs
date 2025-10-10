---
title: wandb verify
---

**Usage**

`wandb verify [OPTIONS]`

**Summary**

Checks and verifies local instance of W&B. W&B checks for:

Checks that the host is not `api.wandb.ai` (host check).

Verifies if the user is logged in correctly using the provided API key
(login check).

Checks that requests are made over HTTPS (secure requests).

Validates the CORS (Cross-Origin Resource Sharing) configuration of the
object store (CORS configuration).

Logs metrics, saves, and downloads files to check if runs are correctly
recorded and accessible (run check).

Saves and downloads artifacts to verify that the artifact storage and
retrieval system is working as expected (artifact check).

Tests the GraphQL endpoint by uploading a file to ensure it can handle
signed URL uploads (GraphQL PUT check).

Checks the ability to send large payloads through the proxy (large payload
check).

Verifies that the installed version of the W&B package is up-to-date and
compatible with the server (W&B version check).

Creates and executes a sweep to ensure that sweep functionality is working
correctly (sweeps check).


**Options**

| **Option** | **Description** |
| :--- | :--- |
| `--host` | Test a specific instance of W&B |



