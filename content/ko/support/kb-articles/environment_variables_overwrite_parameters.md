---
title: 환경 변수는 wandb.init()에 전달된 파라미터를 덮어쓰나요?
menu:
  support:
    identifier: ko-support-kb-articles-environment_variables_overwrite_parameters
support:
- 환경 변수
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init`에 전달된 인수는 환경 변수를 우선하여 적용합니다. 환경 변수가 설정되지 않은 경우 시스템 기본값이 아닌 기본 디렉토리를 설정하려면, `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))`를 사용하세요.