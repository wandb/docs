# string

## Chainable Ops
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

두 값의 불일치를 결정합니다.

| Argument |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값. |
| `rhs` | 비교할 두 번째 값. |

#### Return Value
두 값이 같지 않은지 여부입니다.

<h3 id="string-add"><code>string-add</code></h3>

두 [string](https://docs.wandb.ai/ref/weave/string)을 연결합니다.

| Argument |  |
| :--- | :--- |
| `lhs` | 첫 번째 [string](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 두 번째 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
연결된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

두 값의 같음을 결정합니다.

| Argument |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값. |
| `rhs` | 비교할 두 번째 값. |

#### Return Value
두 값이 같은지 여부입니다.

<h3 id="string-append"><code>string-append</code></h3>

[suffix](https://docs.wandb.ai/ref/weave/string)를 [string](https://docs.wandb.ai/ref/weave/string)에 추가합니다.

| Argument |  |
| :--- | :--- |
| `str` | 추가할 [string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 추가할 접미사 |

#### Return Value
접미사가 추가된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)에 부분 문자열이 포함되어 있는지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 확인할 부분 문자열 |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)에 부분 문자열이 포함되어 있는지 여부입니다.

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 특정 접미사로 끝나는지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 확인할 접미사 |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 접미사로 끝나는지 여부입니다.

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) 내 부분 문자열의 모든 발생을 찾습니다.

| Argument |  |
| :--- | :--- |
| `str` | 부분 문자열의 발생을 찾을 [string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 찾을 부분 문자열 |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string) 내 부분 문자열의 _list_ 인덱스입니다.

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 영숫자인지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 영숫자인지 여부입니다.

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 알파벳 문자인지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 알파벳 문자인지 여부입니다.

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 숫자인지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 숫자인지 여부입니다.

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

앞쪽 공백을 제거합니다.

| Argument |  |
| :--- | :--- |
| `str` | 공백을 제거할 [string](https://docs.wandb.ai/ref/weave/string). |

#### Return Value
공백이 제거된 [string](https://docs.wandb.ai/ref/weave/string).

<h3 id="string-len"><code>string-len</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)의 길이를 반환합니다.

| Argument |  |
| :--- | :--- |
| `str` | 길이를 확인할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)의 길이입니다.

<h3 id="string-lower"><code>string-lower</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)을 소문자로 변환합니다.

| Argument |  |
| :--- | :--- |
| `str` | 소문자로 변환할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
소문자로 변환된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)을 _list_ 형태의 [strings](https://docs.wandb.ai/ref/weave/string)로 나눕니다.

| Argument |  |
| :--- | :--- |
| `str` | 나눌 [string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 나눌 구분자 |

#### Return Value
구분자 이전의 [string](https://docs.wandb.ai/ref/weave/string), 구분자, 구분자 이후의 [string](https://docs.wandb.ai/ref/weave/string)인 _list_ 형태의 [strings](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)에 접두사를 추가합니다.

| Argument |  |
| :--- | :--- |
| `str` | 접두사를 추가할 [string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 추가할 접두사 |

#### Return Value
접두사가 추가된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

뒤쪽 공백을 제거합니다.

| Argument |  |
| :--- | :--- |
| `str` | 공백을 제거할 [string](https://docs.wandb.ai/ref/weave/string). |

#### Return Value
공백이 제거된 [string](https://docs.wandb.ai/ref/weave/string).

<h3 id="string-replace"><code>string-replace</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) 내 부분 문자열의 모든 발생을 다른 부분 문자열로 대체합니다.

| Argument |  |
| :--- | :--- |
| `str` | 내용물을 대체할 [string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 대체할 부분 문자열 |
| `newSub` | 이전 부분 문자열을 대체할 새로운 부분 문자열 |

#### Return Value
대체된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-slice"><code>string-slice</code></h3>

시작 및 종료 인덱스를 기준으로 [string](https://docs.wandb.ai/ref/weave/string)을 부분 문자열로 슬라이스합니다.

| Argument |  |
| :--- | :--- |
| `str` | 슬라이스할 [string](https://docs.wandb.ai/ref/weave/string) |
| `begin` | 부분 문자열의 시작 인덱스 |
| `end` | 부분 문자열의 종료 인덱스 |

#### Return Value
부분 문자열

<h3 id="string-split"><code>string-split</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)을 여러 [strings](https://docs.wandb.ai/ref/weave/string) _list_로 나눕니다.

| Argument |  |
| :--- | :--- |
| `str` | 나눌 [string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 나눌 구분자 |

#### Return Value
_multiple_ [strings](https://docs.wandb.ai/ref/weave/string)의 _list_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 특정 접두사로 시작하는지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 확인할 접두사 |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 접두사로 시작하는지 여부입니다.

<h3 id="string-strip"><code>string-strip</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)의 양쪽 끝에서 공백을 제거합니다.

| Argument |  |
| :--- | :--- |
| `str` | 공백을 제거할 [string](https://docs.wandb.ai/ref/weave/string). |

#### Return Value
공백이 제거된 [string](https://docs.wandb.ai/ref/weave/string).

<h3 id="string-upper"><code>string-upper</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)을 대문자로 변환합니다.

| Argument |  |
| :--- | :--- |
| `str` | 대문자로 변환할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
대문자로 변환된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

두 [strings](https://docs.wandb.ai/ref/weave/string) 간의 Levenshtein 거리를 계산합니다.

| Argument |  |
| :--- | :--- |
| `str1` | 첫 번째 [string](https://docs.wandb.ai/ref/weave/string). |
| `str2` | 두 번째 [string](https://docs.wandb.ai/ref/weave/string). |

#### Return Value
두 [strings](https://docs.wandb.ai/ref/weave/string) 간의 Levenshtein 거리입니다.


## List Ops
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

두 값의 불일치를 결정합니다.

| Argument |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값. |
| `rhs` | 비교할 두 번째 값. |

#### Return Value
두 값이 같지 않은지 여부입니다.

<h3 id="string-add"><code>string-add</code></h3>

두 [string](https://docs.wandb.ai/ref/weave/string)을 연결합니다.

| Argument |  |
| :--- | :--- |
| `lhs` | 첫 번째 [string](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 두 번째 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
연결된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

두 값의 같음을 결정합니다.

| Argument |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값. |
| `rhs` | 비교할 두 번째 값. |

#### Return Value
두 값이 같은지 여부입니다.

<h3 id="string-append"><code>string-append</code></h3>

[suffix](https://docs.wandb.ai/ref/weave/string)를 [string](https://docs.wandb.ai/ref/weave/string)에 추가합니다.

| Argument |  |
| :--- | :--- |
| `str` | 추가할 [string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 추가할 접미사 |

#### Return Value
접미사가 추가된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)에 부분 문자열이 포함되어 있는지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 확인할 부분 문자열 |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)에 부분 문자열이 포함되어 있는지 여부입니다.

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 특정 접미사로 끝나는지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 확인할 접미사 |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 접미사로 끝나는지 여부입니다.

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) 내 부분 문자열의 모든 발생을 찾습니다.

| Argument |  |
| :--- | :--- |
| `str` | 부분 문자열의 발생을 찾을 [string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 찾을 부분 문자열 |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string) 내 부분 문자열의 _list_ 인덱스입니다.

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 영숫자인지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 영숫자인지 여부입니다.

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 알파벳 문자인지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 알파벳 문자인지 여부입니다.

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 숫자인지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 숫자인지 여부입니다.

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

앞쪽 공백을 제거합니다.

| Argument |  |
| :--- | :--- |
| `str` | 공백을 제거할 [string](https://docs.wandb.ai/ref/weave/string). |

#### Return Value
공백이 제거된 [string](https://docs.wandb.ai/ref/weave/string).

<h3 id="string-len"><code>string-len</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)의 길이를 반환합니다.

| Argument |  |
| :--- | :--- |
| `str` | 길이를 확인할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)의 길이입니다.

<h3 id="string-lower"><code>string-lower</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)을 소문자로 변환합니다.

| Argument |  |
| :--- | :--- |
| `str` | 소문자로 변환할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
소문자로 변환된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)을 _list_ 형태의 [strings](https://docs.wandb.ai/ref/weave/string)로 나눕니다.

| Argument |  |
| :--- | :--- |
| `str` | 나눌 [string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 나눌 구분자 |

#### Return Value
구분자 이전의 [string](https://docs.wandb.ai/ref/weave/string), 구분자, 구분자 이후의 [string](https://docs.wandb.ai/ref/weave/string)인 _list_ 형태의 [strings](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)에 접두사를 추가합니다.

| Argument |  |
| :--- | :--- |
| `str` | 접두사를 추가할 [string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 추가할 접두사 |

#### Return Value
접두사가 추가된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

뒤쪽 공백을 제거합니다.

| Argument |  |
| :--- | :--- |
| `str` | 공백을 제거할 [string](https://docs.wandb.ai/ref/weave/string). |

#### Return Value
공백이 제거된 [string](https://docs.wandb.ai/ref/weave/string).

<h3 id="string-replace"><code>string-replace</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) 내 부분 문자열의 모든 발생을 다른 부분 문자열로 대체합니다.

| Argument |  |
| :--- | :--- |
| `str` | 내용물을 대체할 [string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 대체할 부분 문자열 |
| `newSub` | 이전 부분 문자열을 대체할 새로운 부분 문자열 |

#### Return Value
대체된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-slice"><code>string-slice</code></h3>

시작 및 종료 인덱스를 기준으로 [string](https://docs.wandb.ai/ref/weave/string)을 부분 문자열로 슬라이스합니다.

| Argument |  |
| :--- | :--- |
| `str` | 슬라이스할 [string](https://docs.wandb.ai/ref/weave/string) |
| `begin` | 부분 문자열의 시작 인덱스 |
| `end` | 부분 문자열의 종료 인덱스 |

#### Return Value
부분 문자열

<h3 id="string-split"><code>string-split</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)을 여러 [strings](https://docs.wandb.ai/ref/weave/string) _list_로 나눕니다.

| Argument |  |
| :--- | :--- |
| `str` | 나눌 [string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 나눌 구분자 |

#### Return Value
_multiple_ [strings](https://docs.wandb.ai/ref/weave/string)의 _list_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)이 특정 접두사로 시작하는지 확인합니다.

| Argument |  |
| :--- | :--- |
| `str` | 확인할 [string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 확인할 접두사 |

#### Return Value
[string](https://docs.wandb.ai/ref/weave/string)이 접두사로 시작하는지 여부입니다.

<h3 id="string-strip"><code>string-strip</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)의 양쪽 끝에서 공백을 제거합니다.

| Argument |  |
| :--- | :--- |
| `str` | 공백을 제거할 [string](https://docs.wandb.ai/ref/weave/string). |

#### Return Value
공백이 제거된 [string](https://docs.wandb.ai/ref/weave/string).

<h3 id="string-upper"><code>string-upper</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)을 대문자로 변환합니다.

| Argument |  |
| :--- | :--- |
| `str` | 대문자로 변환할 [string](https://docs.wandb.ai/ref/weave/string) |

#### Return Value
대문자로 변환된 [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

두 [strings](https://docs.wandb.ai/ref/weave/string) 간의 Levenshtein 거리를 계산합니다.

| Argument |  |
| :--- | :--- |
| `str1` | 첫 번째 [string](https://docs.wandb.ai/ref/weave/string). |
| `str2` | 두 번째 [string](https://docs.wandb.ai/ref/weave/string). |

#### Return Value
두 [strings](https://docs.wandb.ai/ref/weave/string) 간의 Levenshtein 거리입니다.