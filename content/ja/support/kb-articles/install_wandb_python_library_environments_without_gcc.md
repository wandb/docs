---
title: gcc のない 環境 で wandb の Python ライブラリ はどのようにインストールできますか？
menu:
  support:
    identifier: ja-support-kb-articles-install_wandb_python_library_environments_without_gcc
support:
- python
toc_hide: true
type: docs
url: /support/:filename
---

`wandb` をインストールする際に、次のエラーが表示される場合:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

事前ビルド済みの wheel から `psutil` を直接インストールしてください。Python のバージョンと OS は [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil) で確認できます。

例: Linux 上の Python 3.8 に `psutil` をインストールする場合:

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil` のインストール後、`pip install wandb` を実行して `wandb` のインストールを完了します。