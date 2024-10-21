---
title: "How do I install the wandb Python library in environments without gcc?"
tags: []
---

### How do I install the wandb Python library in environments without gcc?
If you try to install `wandb` and see this error:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

You can install `psutil` directly from a pre-built wheel. Find your Python version and OS here: [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil)

For example, to install `psutil` on Python 3.8 in Linux:

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

After `psutil` has been installed, you can install wandb with `pip install wandb`.