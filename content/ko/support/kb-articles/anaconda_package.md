---
title: Weights and Biases용 anaconda 패키지가 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-anaconda_package
support:
- Python
toc_hide: true
type: docs
url: /support/:filename
---

Anaconda 패키지는 `pip` 또는 `conda`를 사용하여 설치할 수 있습니다. `conda`를 사용할 경우, [conda-forge](https://conda-forge.org) 채널에서 패키지를 받으세요.

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# conda 환경 생성
conda create -n wandb-env python=3.8 anaconda
# 환경 활성화
conda activate wandb-env
# pip으로 wandb 설치
pip install wandb
```
{{% /tab %}}
{{% tab "conda" %}}
```shell
conda activate myenv
conda install wandb --channel conda-forge
```
{{% /tab %}}
{{< /tabpane >}}

설치 관련 문제가 발생하면 Anaconda의 [패키지 관리 가이드](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)를 참고하세요.