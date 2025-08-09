---
title: 왜 W&B Inference 에서 내 국가 또는 지역이 지원되지 않는다고 나오나요?
menu:
  support:
    identifier: ko-support-kb-articles-inference_geographic_restriction_403
support:
- 추론
toc_hide: true
type: docs
url: /support/:filename
---

"Country, region, or territory not supported"라는 메시지와 함께 403 오류가 발생하면, 지원되지 않는 위치에서 W&B Inference에 엑세스하려고 했다는 의미입니다.

## 왜 이런 현상이 발생하나요

W&B Inference는 컴플라이언스 및 규제 요건으로 인해 지리적 제한이 있습니다. 이 서비스는 지원되는 지리적 위치에서만 사용할 수 있습니다.

## 할 수 있는 일

1. **이용 약관 확인하기**
   - 현재 지원되는 위치 목록은 [이용 약관](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions)에서 확인하세요.

2. **지원되는 위치에서 사용하기**
   - 지원되는 국가나 지역에 있을 때 서비스에 엑세스하세요.
   - 지원되는 위치에 있는 조직의 리소스를 활용하는 것도 고려해 보세요.

3. **계정 팀에 문의하기**
   - 엔터프라이즈 고객은 계정 담당자와 옵션을 논의할 수 있습니다.
   - 일부 조직은 별도의 특별 약정을 가지고 있을 수 있습니다.

## 오류 상세 내용

이 오류가 표시될 때:
```
{
  "error": {
    "code": 403,
    "message": "Country, region, or territory not supported"
  }
}
```

이 메시지는 API 요청 시점의 IP 어드레스 위치에 따라 결정됩니다.