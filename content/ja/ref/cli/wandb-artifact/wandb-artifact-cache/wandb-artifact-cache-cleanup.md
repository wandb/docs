---
title: wandb アーティファクト キャッシュのクリーンアップ
menu:
  reference:
    identifier: ja-ref-cli-wandb-artifact-wandb-artifact-cache-wandb-artifact-cache-cleanup
---

**使い方**

`wandb artifact cache cleanup [OPTIONS] TARGET_SIZE`

**概要**

Artifacts キャッシュから使用頻度の低いファイルをクリーンアップします


**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--remove-temp / --no-remove-temp` | 一時ファイルを削除します |