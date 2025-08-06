---
title: W&B 클라이언트는 Python 2를 지원하나요?
menu:
  support:
    identifier: ko-support-kb-articles-client_support_python_2
support:
- Python
toc_hide: true
type: docs
url: /support/:filename
---

W&B 클라이언트 라이브러리는 0.10 버전까지 Python 2.7과 Python 3 모두를 지원했습니다. 하지만 Python 2의 공식 지원이 종료됨에 따라, 0.11 버전부터는 Python 2.7 지원이 중단되었습니다. Python 2.7 환경에서 `pip install --upgrade wandb` 명령어를 실행하면 0.10.x 시리즈의 최신 릴리스만 설치됩니다. 0.10.x 시리즈는 보안 및 치명적인 버그 수정만 제공되며, Python 2.7을 지원하는 마지막 0.10.x 버전은 0.10.33입니다.