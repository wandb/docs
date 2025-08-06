---
title: gcc がない環境で wandb Python ライブラリをインストールするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-install_wandb_python_library_environments_without_gcc
support:
- 'python

  '
toc_hide: true
type: docs
url: /support/:filename
---

`wandb` のインストール時に次のようなエラーが表示された場合：

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

`psutil` を事前ビルド済みホイールから直接インストールしてください。ご自身の Python バージョンとオペレーティングシステムを [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil) で確認します。

例えば、Linux で Python 3.8 を使用している場合の `psutil` インストール方法は以下の通りです。

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil` のインストールが完了したら、`pip install wandb` を実行して `wandb` のインストールを完了してください。