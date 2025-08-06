---
title: 리포트에 CSV 업로드하기
menu:
  support:
    identifier: ko-support-kb-articles-upload_csv_report
support:
- 리포트
toc_hide: true
type: docs
url: /support/:filename
---

CSV를 리포트에 업로드하려면 `wandb.Table` 포맷을 사용하세요. Python 스크립트에서 CSV를 불러오고, 이를 `wandb.Table` 오브젝트로 로그합니다. 이 작업을 통해 데이터가 리포트에서 테이블 형식으로 보여집니다.