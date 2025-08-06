---
title: gcc なしの環境で wandb Python ライブラリをインストールするにはどうすれば良いですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- パイソン
---

もし `wandb` のインストール時に次のようなエラーが表示された場合：

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

`psutil` を事前にビルドされた wheel から直接インストールしてください。お使いの Python バージョンとオペレーティングシステムを [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil) で確認してください。

例えば、Linux の Python 3.8 で `psutil` をインストールする場合は次のようになります。

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil` のインストールが完了したら、`pip install wandb` を実行し `wandb` のインストールを完了してください。