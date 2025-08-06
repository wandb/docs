---
title: 스윕이 실행 중일 때 Python 파일을 수정하면 어떻게 되나요?
menu:
  support:
    identifier: ko-support-kb-articles-what_happens_if_i_edit_my_python_files_while_a_sweep_is_running
support:
- 스윕
toc_hide: true
type: docs
url: /support/:filename
---

스윕이 실행 중일 때:
- 스윕에서 사용하는 `train.py` 스크립트가 변경되어도, 스윕은 원래의 `train.py`를 계속 사용합니다.
- 하지만 `train.py` 스크립트가 참조하는 파일(예: `helper.py` 스크립트 내의 헬퍼 함수)이 변경될 경우, 스윕은 업데이트된 `helper.py`를 사용하기 시작합니다.