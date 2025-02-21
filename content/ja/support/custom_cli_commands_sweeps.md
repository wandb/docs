---
title: How do I use custom CLI commands with sweeps?
menu:
  support:
    identifier: ja-support-custom_cli_commands_sweeps
tags:
- sweeps
toc_hide: true
type: docs
---

W&B Sweeps は、トレーニング設定がコマンドライン 引数を渡す場合、カスタム CLI コマンドで使用できます。

以下の例では、ユーザーが `train.py` という名前の Python スクリプトをトレーニング し、スクリプトが解析する値を指定する bash ターミナルを示すコード スニペットを示しています。

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

カスタム コマンドを実装するには、YAML ファイルの `command` キーを変更します。前の例に基づいて、設定は次のようになります。

```yaml
program:
  train.py
method: grid
parameters:
  batch_size:
    value: 8
  lr:
    value: 0.0001
command:
  - ${env}
  - python
  - ${program}
  - "-b"
  - your-training-config
  - ${args}
```

`${args}` キーは、sweep configuration のすべての パラメータ に展開され、`argparse` 用に `--param1 value1 --param2 value2` としてフォーマットされます。

`argparse` 以外の追加の 引数 については、以下を実装します。

```python
import argparse
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
環境によっては、`python` が Python 2 を指している場合があります。Python 3 の呼び出しを確実にするには、コマンド 設定で `python3` を使用します。

```yaml
program:
  script.py
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
{{% /alert %}}
