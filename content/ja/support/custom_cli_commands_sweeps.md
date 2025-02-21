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

W&B Sweeps は、トレーニング設定がコマンドライン引数を介する場合、カスタム CLI コマンドで使用できます。

以下の例では、ユーザーが `train.py` という名前の Python スクリプトをトレーニングし、スクリプトが解析する値を提供する bash ターミナルを示すコードスニペットを示します。

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

カスタムコマンドを実装するには、YAML ファイル内の `command` キーを修正します。前の例に基づくと、設定は次のようになります。

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

`${args}` キーは、すべてのパラメータを sweep configuration に展開し、`argparse` の形式で `--param1 value1 --param2 value2` としてフォーマットされます。

`argparse` 以外の引数を追加する場合は、次を実装します。

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
環境によっては、`python` が Python 2 を指すことがあります。Python 3 を呼び出すには、コマンド設定で `python3` を使用してください:

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