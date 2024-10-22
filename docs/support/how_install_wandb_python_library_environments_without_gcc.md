---
title: "How do I install the wandb Python library in environments without gcc?"
tags:
   - None
---
If an error occurs when installing `wandb` that states:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

Install `psutil` directly from a pre-built wheel. Determine your Python version and operating system at [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil).

For example, to install `psutil` on Python 3.8 in Linux:

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

After installing `psutil`, run `pip install wandb` to complete the installation of `wandb`.