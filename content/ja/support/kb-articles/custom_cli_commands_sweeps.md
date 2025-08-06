---
title: スイープでカスタム CLI コマンドを使うにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
---

W&B Sweeps は、トレーニング設定をコマンドライン引数として渡す場合、カスタム CLI コマンドでも利用できます。

以下の例では、bash ターミナルで Users が `train.py` という Python スクリプトをトレーニングする様子をコードスニペットで示しています。スクリプトは渡された値をパースします。

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

カスタムコマンドを実装するには、YAML ファイルの `command` キーを修正します。前述の例に基づくと、設定は以下のようになります。

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

`${args}` キーは、sweep configuration のすべてのパラメータを、`argparse` 用（`--param1 value1 --param2 value2` の形式）に展開します。

`argparse` 以外の追加の引数が必要な場合は、次のように実装してください。

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
環境によっては、`python` が Python 2 を指す場合があります。Python 3 を必ず呼び出すには、コマンド設定で `python3` を使用してください。

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