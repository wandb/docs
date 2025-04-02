---
title: How do I use custom CLI commands with sweeps?
menu:
  support:
    identifier: ja-support-kb-articles-custom_cli_commands_sweeps
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

W&B Sweeps は、トレーニング設定がコマンドライン引数を渡す場合、カスタム CLI コマンドで使用できます。

以下の例では、ユーザーが `train.py` という Python スクリプトをトレーニングし、スクリプトが解析する値を指定する bash ターミナルを示すコードスニペットを示しています。

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

カスタムコマンドを実装するには、YAML ファイルの `command` キーを修正します。前の例に基づくと、設定は次のようになります。

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

`${args}` キーは、sweep configuration 内のすべてのパラメータに展開され、`argparse` 用に `--param1 value1 --param2 value2` としてフォーマットされます。

`argparse` 以外の追加の 引数 については、以下を実装します。

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
環境によっては、`python` が Python 2 を指す場合があります。Python 3 の起動を確実にするには、コマンド設定で `python3` を使用します。

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
