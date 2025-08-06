---
title: wandb アーティファクトキャッシュのクリーンアップ
---

**使用方法**

`wandb artifact cache cleanup [OPTIONS] TARGET_SIZE`

**概要**

Artifacts キャッシュからあまり使われていないファイルをクリーンアップします。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--remove-temp / --no-remove-temp` | 一時ファイルを削除します |