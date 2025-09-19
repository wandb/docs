---
title: wandb local
---

Start a local W&B container (deprecated, see wandb server --help)

## Usage

```bash
wandb local [OPTIONS]
```

## Options

| Option | Description |
| :--- | :--- |
| `--port`, `-p` | The host port to bind W&B local on (default: 8080) |
| `--env`, `-e` | Env vars to pass to wandb/local (default: []) |
| `--daemon` | Run or don't run in daemon mode (default: True) |
| `--upgrade` | Upgrade to the most recent version (default: False) |
| `--edge` | Run the bleeding edge (default: False) |
