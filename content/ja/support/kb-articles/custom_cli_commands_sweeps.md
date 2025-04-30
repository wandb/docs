---
title: スイープでカスタム CLI コマンドを使用するにはどうすれば良いですか？
menu:
  support:
    identifier: ja-support-kb-articles-custom_cli_commands_sweeps
support:
- sweep
toc_hide: true
type: docs
url: /ja/support/:filename
---

W&B スイープは、トレーニング設定がコマンドライン引数を通過する場合、カスタム CLI コマンドと共に使用できます。

以下の例では、コードスニペットが bash ターミナルを示しており、ユーザーが `train.py` という名前の Python スクリプトをトレーニングし、スクリプトが解析する値を提供しています。

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

カスタムコマンドを実装するには、YAML ファイル内の `command` キーを修正します。前の例に基づくと、設定は次のようになります:

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

`${args}` キーは、スイープ設定内のすべてのパラメータを `argparse` 用に `--param1 value1 --param2 value2` という形式に展開します。

`argparse` 以外の追加の引数については、次を実装してください:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
環境によっては、`python` が Python 2 を指す場合があります。Python 3 を呼び出すには、コマンド設定で `python3` を使用してください。

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