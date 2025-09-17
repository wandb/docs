---
title: Sweeps でカスタム CLI コマンドを使うには？
menu:
  support:
    identifier: ja-support-kb-articles-custom_cli_commands_sweeps
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング の 設定が コマンドライン 引数を 受け取れる 場合は、W&B Sweeps を カスタム CLI コマンド と 併用できます。

次の 例では、bash ターミナル で ユーザー が `train.py` という Python スクリプト の トレーニング を 実行し、スクリプト が 解釈する 値 を 渡す 様子を コードスニペット で 示します:

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

カスタム コマンド を 実装するには、YAML ファイル の `command` キー を 変更します。先ほどの 例に 基づく 設定 は 次の とおりです:

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

`${args}` キー は、sweep configuration 内の すべての パラメータ を 展開し、`argparse` 用に `--param1 value1 --param2 value2` の 形式に 整形します。

`argparse` 以外で 追加の 引数 を 受け取りたい 場合は、次の ように 実装します:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
環境 によっては、`python` が Python 2 を 指す 場合が あります。Python 3 を 確実に 呼び出す には、コマンド 設定 で `python3` を 使用してください:

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