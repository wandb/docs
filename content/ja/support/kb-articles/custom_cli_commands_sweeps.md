---
title: スイープでカスタム CLI コマンドを使うにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-custom_cli_commands_sweeps
support:
- スイープ
toc_hide: true
type: docs
url: /support/:filename
---

W&B Sweeps は、トレーニング設定がコマンドライン引数で渡される場合、カスタム CLI コマンドと組み合わせて利用できます。

下記の例では、ユーザーが `train.py` という Python スクリプトをターミナルでトレーニングし、スクリプトが値を受け取ってパースする様子をコードスニペットで示しています。

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

カスタムコマンドを実装するには、YAML ファイル内の `command` キーを変更します。先ほどの例をもとにした設定は下記のようになります。

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

`${args}` キーは sweep configuration 内のすべてのパラメータを自動展開し、`argparse` 向けの `--param1 value1 --param2 value2` 形式で渡されます。

`argparse` とは別の追加引数を扱いたい場合、下記のように記述します。

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
環境によっては、`python` コマンドが Python 2 を指す場合があります。Python 3 を確実に使用するには、コマンド設定で `python3` を利用してください。

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