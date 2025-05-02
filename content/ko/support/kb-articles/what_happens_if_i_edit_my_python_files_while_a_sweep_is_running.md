---
title: What happens if I edit my Python files while a sweep is running?
menu:
  support:
    identifier: ko-support-kb-articles-what_happens_if_i_edit_my_python_files_while_a_sweep_is_running
support:
- sweeps
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: what_happens_if_i_edit_my_python_files_while_a_sweep_is_running
---
스윕이 실행되는 동안:
- 스윕에서 사용하는 `train.py` 스크립트가 변경되면 스윕은 원래 `train.py`를 계속 사용합니다.
- `train.py` 스크립트가 참조하는 파일(예: `helper.py` 스크립트의 helper 함수)이 변경되면 스윕은 업데이트된 `helper.py`를 사용하기 시작합니다.
