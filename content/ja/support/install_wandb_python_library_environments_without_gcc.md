---
title: How do I install the wandb Python library in environments without gcc?
menu:
  support:
    identifier: ja-support-install_wandb_python_library_environments_without_gcc
tags:
- python
toc_hide: true
type: docs
---

`wandb` をインストールするときに次のようなエラーが発生した場合:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

事前にビルドされたホイールから `psutil` を直接インストールしてください。あなたの Python バージョンとオペレーティングシステムを [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil) で確認してください。

例えば、Linux 上で Python 3.8 に `psutil` をインストールするには:

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil` のインストール後、`wandb` のインストールを完了するために `pip install wandb` を実行してください。