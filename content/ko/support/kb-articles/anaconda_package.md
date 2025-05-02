---
title: Is there an anaconda package for Weights and Biases?
menu:
  support:
    identifier: ko-support-kb-articles-anaconda_package
support:
- python
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: anaconda_package
---
`pip` 또는 `conda`를 사용하여 설치할 수 있는 Anaconda 패키지가 있습니다. `conda`의 경우 [conda-forge](https://conda-forge.org) 채널에서 패키지를 받으세요.

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# conda 환경 생성
conda create -n wandb-env python=3.8 anaconda
# 환경 활성화
conda activate wandb-env
# pip를 사용하여 wandb 설치
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

설치 문제에 대해서는 Anaconda의 [패키지 관리 관련 문서](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)를 참조하여 도움을 받으세요.
