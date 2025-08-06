---
title: 문자열
menu:
  reference:
    identifier: ko-ref-query-panel-string
---

## 체이너블 연산자 (Chainable Ops)
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

두 값이 서로 다른지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값 |
| `rhs` | 비교할 두 번째 값 |

#### 반환 값
두 값이 서로 다른지 여부

<h3 id="string-add"><code>string-add</code></h3>

두 [string](string.md)을 이어 붙입니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [string](string.md) |
| `rhs` | 두 번째 [string](string.md) |

#### 반환 값
이어 붙여진 [string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

두 값이 서로 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값 |
| `rhs` | 비교할 두 번째 값 |

#### 반환 값
두 값이 같은지 여부

<h3 id="string-append"><code>string-append</code></h3>

[string](string.md)에 접미사를 추가합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 접미사를 추가할 [string](string.md) |
| `suffix` | 추가할 접미사 |

#### 반환 값
접미사가 추가된 [string](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](string.md)에 특정 부분 문자열이 포함되어 있는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |
| `sub` | 찾을 부분 문자열 |

#### 반환 값
[string](string.md)에 부분 문자열이 포함되어 있는지 여부

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](string.md)이 특정 접미사로 끝나는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |
| `suffix` | 확인할 접미사 |

#### 반환 값
[string](string.md)이 해당 접미사로 끝나는지 여부

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](string.md) 내에서 부분 문자열이 등장하는 모든 위치를 찾습니다.

| 인수 |  |
| :--- | :--- |
| `str` | 부분 문자열의 위치를 찾을 [string](string.md) |
| `sub` | 찾을 부분 문자열 |

#### 반환 값
[string](string.md) 내에서 부분 문자열이 등장하는 인덱스들의 _list_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](string.md)이 영숫자(알파벳 또는 숫자)로만 이루어져 있는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |

#### 반환 값
[string](string.md)이 영숫자인지 여부

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](string.md)이 알파벳으로만 이루어져 있는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |

#### 반환 값
[string](string.md)이 알파벳으로만 이루어져 있는지 여부

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](string.md)이 숫자로만 이루어져 있는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |

#### 반환 값
[string](string.md)이 숫자인지 여부

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

앞쪽 공백을 제거합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 앞쪽 공백을 제거할 [string](string.md) |

#### 반환 값
공백이 제거된 [string](string.md)

<h3 id="string-len"><code>string-len</code></h3>

[string](string.md)의 길이를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 길이를 확인할 [string](string.md) |

#### 반환 값
[string](string.md)의 길이

<h3 id="string-lower"><code>string-lower</code></h3>

[string](string.md)을 소문자로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 소문자로 변환할 [string](string.md) |

#### 반환 값
소문자로 변환된 [string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](string.md)을 _list_로 파티션합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 분할할 [string](string.md) |
| `sep` | 기준이 될 구분자 |

#### 반환 값
_list_ 형태의 [string](string.md): 구분자 앞의 문자열, 구분자, 구분자 뒤의 문자열

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](string.md) 앞에 접두사를 붙입니다.

| 인수 |  |
| :--- | :--- |
| `str` | 접두사를 붙일 [string](string.md) |
| `prefix` | 붙일 접두사 |

#### 반환 값
접두사가 붙은 [string](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

뒤쪽 공백을 제거합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 뒤쪽 공백을 제거할 [string](string.md) |

#### 반환 값
공백이 제거된 [string](string.md)

<h3 id="string-replace"><code>string-replace</code></h3>

[string](string.md)에 포함된 특정 부분 문자열을 모두 다른 부분 문자열로 교체합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 내용이 교체될 [string](string.md) |
| `sub` | 교체할 부분 문자열 |
| `newSub` | 새로운 부분 문자열 |

#### 반환 값
부분 문자열이 교체된 [string](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

시작과 끝 인덱스를 기준으로 [string](string.md)에서 부분 문자열을 추출합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 분할할 [string](string.md) |
| `begin` | 시작 인덱스 |
| `end` | 종료 인덱스 |

#### 반환 값
부분 문자열

<h3 id="string-split"><code>string-split</code></h3>

[string](string.md)을 _list_ 형태의 [string](string.md)들로 분할합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 분할할 [string](string.md) |
| `sep` | 분할 기준이 될 구분자 |

#### 반환 값
_list_ 형태의 [string](string.md) 목록

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](string.md)이 특정 접두사로 시작하는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |
| `prefix` | 확인할 접두사 |

#### 반환 값
[string](string.md)이 해당 접두사로 시작하는지 여부

<h3 id="string-strip"><code>string-strip</code></h3>

[string](string.md)의 양쪽 끝에 있는 공백을 모두 제거합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 공백을 제거할 [string](string.md) |

#### 반환 값
공백이 제거된 [string](string.md)

<h3 id="string-upper"><code>string-upper</code></h3>

[string](string.md)을 대문자로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 대문자로 변환할 [string](string.md) |

#### 반환 값
대문자로 변환된 [string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

두 [string](string.md) 사이의 Levenshtein 거리를 계산합니다.

| 인수 |  |
| :--- | :--- |
| `str1` | 첫 번째 [string](string.md) |
| `str2` | 두 번째 [string](string.md) |

#### 반환 값
두 [string](string.md) 사이의 Levenshtein 거리


## 리스트 연산자 (List Ops)
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

두 값이 서로 다른지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값 |
| `rhs` | 비교할 두 번째 값 |

#### 반환 값
두 값이 서로 다른지 여부

<h3 id="string-add"><code>string-add</code></h3>

두 [string](string.md)을 이어 붙입니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [string](string.md) |
| `rhs` | 두 번째 [string](string.md) |

#### 반환 값
이어 붙여진 [string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

두 값이 서로 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값 |
| `rhs` | 비교할 두 번째 값 |

#### 반환 값
두 값이 같은지 여부

<h3 id="string-append"><code>string-append</code></h3>

[string](string.md)에 접미사를 추가합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 접미사를 추가할 [string](string.md) |
| `suffix` | 추가할 접미사 |

#### 반환 값
접미사가 추가된 [string](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](string.md)에 특정 부분 문자열이 포함되어 있는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |
| `sub` | 찾을 부분 문자열 |

#### 반환 값
[string](string.md)에 부분 문자열이 포함되어 있는지 여부

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](string.md)이 특정 접미사로 끝나는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |
| `suffix` | 확인할 접미사 |

#### 반환 값
[string](string.md)이 해당 접미사로 끝나는지 여부

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](string.md) 내에서 부분 문자열이 등장하는 모든 위치를 찾습니다.

| 인수 |  |
| :--- | :--- |
| `str` | 부분 문자열의 위치를 찾을 [string](string.md) |
| `sub` | 찾을 부분 문자열 |

#### 반환 값
[string](string.md) 내에서 부분 문자열이 등장하는 인덱스들의 _list_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](string.md)이 영숫자(알파벳 또는 숫자)로만 이루어져 있는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |

#### 반환 값
[string](string.md)이 영숫자인지 여부

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](string.md)이 알파벳으로만 이루어져 있는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |

#### 반환 값
[string](string.md)이 알파벳으로만 이루어져 있는지 여부

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](string.md)이 숫자로만 이루어져 있는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |

#### 반환 값
[string](string.md)이 숫자인지 여부

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

앞쪽 공백을 제거합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 앞쪽 공백을 제거할 [string](string.md) |

#### 반환 값
공백이 제거된 [string](string.md)

<h3 id="string-len"><code>string-len</code></h3>

[string](string.md)의 길이를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 길이를 확인할 [string](string.md) |

#### 반환 값
[string](string.md)의 길이

<h3 id="string-lower"><code>string-lower</code></h3>

[string](string.md)을 소문자로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 소문자로 변환할 [string](string.md) |

#### 반환 값
소문자로 변환된 [string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](string.md)을 _list_로 파티션합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 분할할 [string](string.md) |
| `sep` | 기준이 될 구분자 |

#### 반환 값
_list_ 형태의 [string](string.md): 구분자 앞의 문자열, 구분자, 구분자 뒤의 문자열

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](string.md) 앞에 접두사를 붙입니다.

| 인수 |  |
| :--- | :--- |
| `str` | 접두사를 붙일 [string](string.md) |
| `prefix` | 붙일 접두사 |

#### 반환 값
접두사가 붙은 [string](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

뒤쪽 공백을 제거합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 뒤쪽 공백을 제거할 [string](string.md) |

#### 반환 값
공백이 제거된 [string](string.md)

<h3 id="string-replace"><code>string-replace</code></h3>

[string](string.md)에 포함된 특정 부분 문자열을 모두 다른 부분 문자열로 교체합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 내용이 교체될 [string](string.md) |
| `sub` | 교체할 부분 문자열 |
| `newSub` | 새로운 부분 문자열 |

#### 반환 값
부분 문자열이 교체된 [string](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

시작과 끝 인덱스를 기준으로 [string](string.md)에서 부분 문자열을 추출합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 분할할 [string](string.md) |
| `begin` | 시작 인덱스 |
| `end` | 종료 인덱스 |

#### 반환 값
부분 문자열

<h3 id="string-split"><code>string-split</code></h3>

[string](string.md)을 _list_ 형태의 [string](string.md)들로 분할합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 분할할 [string](string.md) |
| `sep` | 분할 기준이 될 구분자 |

#### 반환 값
_list_ 형태의 [string](string.md) 목록

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](string.md)이 특정 접두사로 시작하는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [string](string.md) |
| `prefix` | 확인할 접두사 |

#### 반환 값
[string](string.md)이 해당 접두사로 시작하는지 여부

<h3 id="string-strip"><code>string-strip</code></h3>

[string](string.md)의 양쪽 끝에 있는 공백을 모두 제거합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 공백을 제거할 [string](string.md) |

#### 반환 값
공백이 제거된 [string](string.md)

<h3 id="string-upper"><code>string-upper</code></h3>

[string](string.md)을 대문자로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 대문자로 변환할 [string](string.md) |

#### 반환 값
대문자로 변환된 [string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

두 [string](string.md) 사이의 Levenshtein 거리를 계산합니다.

| 인수 |  |
| :--- | :--- |
| `str1` | 첫 번째 [string](string.md) |
| `str2` | 두 번째 [string](string.md) |

#### 반환 값
두 [string](string.md) 사이의 Levenshtein 거리