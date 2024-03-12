
# 숫자

## 체인 가능한 연산
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

두 값의 불일치를 판별합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값. |
| `rhs` | 비교할 두 번째 값. |

#### 반환 값
두 값이 같지 않은지 여부.

<h3 id="number-modulo"><code>number-modulo</code></h3>

숫자를 다른 숫자로 나누고 나머지를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 숫자 |
| `rhs` | 나눌 숫자 |

#### 반환 값
두 숫자의 모듈로

<h3 id="number-mult"><code>number-mult</code></h3>

두 숫자를 곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 숫자 |
| `rhs` | 두 번째 숫자 |

#### 반환 값
두 숫자의 곱

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

숫자를 지수로 거듭제곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 기본 숫자 |
| `rhs` | 지수 숫자 |

#### 반환 값
기본 숫자가 n 제곱된 값

<h3 id="number-add"><code>number-add</code></h3>

두 숫자를 더합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 숫자 |
| `rhs` | 두 번째 숫자 |

#### 반환 값
두 숫자의 합

<h3 id="number-sub"><code>number-sub</code></h3>

한 숫자에서 다른 숫자를 뺍니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 빼고자 하는 숫자 |
| `rhs` | 빼질 숫자 |

#### 반환 값
두 숫자의 차이

<h3 id="number-div"><code>number-div</code></h3>

한 숫자를 다른 숫자로 나눕니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 숫자 |
| `rhs` | 나눌 숫자 |

#### 반환 값
두 숫자의 몫

<h3 id="number-less"><code>number-less</code></h3>

한 숫자가 다른 숫자보다 작은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 숫자 |
| `rhs` | 비교 대상 숫자 |

#### 반환 값
첫 번째 숫자가 두 번째 숫자보다 작은지 여부

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

한 숫자가 다른 숫자보다 작거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 숫자 |
| `rhs` | 비교 대상 숫자 |

#### 반환 값
첫 번째 숫자가 두 번째 숫자보다 작거나 같은지 여부

<h3 id="number-equal"><code>number-equal</code></h3>

두 값의 동등성을 판별합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값. |
| `rhs` | 비교할 두 번째 값. |

#### 반환 값
두 값이 같은지 여부.

<h3 id="number-greater"><code>number-greater</code></h3>

한 숫자가 다른 숫자보다 큰지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 숫자 |
| `rhs` | 비교 대상 숫자 |

#### 반환 값
첫 번째 숫자가 두 번째 숫자보다 큰지 여부

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

한 숫자가 다른 숫자보다 크거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 숫자 |
| `rhs` | 비교 대상 숫자 |

#### 반환 값
첫 번째 숫자가 두 번째 숫자보다 크거나 같은지 여부

<h3 id="number-negate"><code>number-negate</code></h3>

숫자의 부호를 반전시킵니다.

| 인수 |  |
| :--- | :--- |
| `val` | 부호를 반전시킬 숫자 |

#### 반환 값
숫자

<h3 id="number-toString"><code>number-toString</code></h3>

숫자를 문자열로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `in` | 변환할 숫자 |

#### 반환 값
숫자의 문자열 표현

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

숫자를 _타임스탬프_로 변환합니다. 31536000000보다 작은 값은 초로, 31536000000000보다 작은 값은 밀리초로, 31536000000000000보다 작은 값은 마이크로초로, 31536000000000000000보다 작은 값은 나노초로 변환됩니다.

| 인수 |  |
| :--- | :--- |
| `val` | 타임스탬프로 변환할 숫자 |

#### 반환 값
타임스탬프

<h3 id="number-abs"><code>number-abs</code></h3>

숫자의 절대값을 계산합니다.

| 인수 |  |
| :--- | :--- |
| `n` | 숫자 |

#### 반환 값
숫자의 절대값

## 리스트 연산
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

두 값의 불일치를 판별합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값. |
| `rhs` | 비교할 두 번째 값. |

#### 반환 값
두 값이 같지 않은지 여부.

<h3 id="number-modulo"><code>number-modulo</code></h3>

숫자를 다른 숫자로 나누고 나머지를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 숫자 |
| `rhs` | 나눌 숫자 |

#### 반환 값
두 숫자의 모듈로

<h3 id="number-mult"><code>number-mult</code></h3>

두 숫자를 곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 숫자 |
| `rhs` | 두 번째 숫자 |

#### 반환 값
두 숫자의 곱

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

숫자를 지수로 거듭제곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 기본 숫자 |
| `rhs` | 지수 숫자 |

#### 반환 값
기본 숫자가 n 제곱된 값

<h3 id="number-add"><code>number-add</code></h3>

두 숫자를 더합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 숫자 |
| `rhs` | 두 번째 숫자 |

#### 반환 값
두 숫자의 합

<h3 id="number-sub"><code>number-sub</code></h3>

한 숫자에서 다른 숫자를 뺍니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 빼고자 하는 숫자 |
| `rhs` | 빼질 숫자 |

#### 반환 값
두 숫자의 차이

<h3 id="number-div"><code>number-div</code></h3>

한 숫자를 다른 숫자로 나눕니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 숫자 |
| `rhs` | 나눌 숫자 |

#### 반환 값
두 숫자의 몫

<h3 id="number-less"><code>number-less</code></h3>

한 숫자가 다른 숫자보다 작은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 숫자 |
| `rhs` | 비교 대상 숫자 |

#### 반환 값
첫 번째 숫자가 두 번째 숫자보다 작은지 여부

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

한 숫자가 다른 숫자보다 작거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 숫자 |
| `rhs` | 비교 대상 숫자 |

#### 반환 값
첫 번째 숫자가 두 번째 숫자보다 작거나 같은지 여부

<h3 id="number-equal"><code>number-equal</code></h3>

두 값의 동등성을 판별합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값. |
| `rhs` | 비교할 두 번째 값. |

#### 반환 값
두 값이 같은지 여부.

<h3 id="number-greater"><code>number-greater</code></h3>

한 숫자가 다른 숫자보다 큰지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 숫자 |
| `rhs` | 비교 대상 숫자 |

#### 반환 값
첫 번째 숫자가 두 번째 숫자보다 큰지 여부

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

한 숫자가 다른 숫자보다 크거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 숫자 |
| `rhs` | 비교 대상 숫자 |

#### 반환 값
첫 번째 숫자가 두 번째 숫자보다 크거나 같은지 여부

<h3 id="number-negate"><code>number-negate</code></h3>

숫자의 부호를 반전시킵니다.

| 인수 |  |
| :--- | :--- |
| `val` | 부호를 반전시킬 숫자 |

#### 반환 값
숫자

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

최대값을 가진 숫자의 인덱스를 찾습니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 최대 숫자의 인덱스를 찾을 숫자 _리스트_ |

#### 반환 값
최대 숫자의 인덱스

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

최소값을 가진 숫자의 인덱스를 찾습니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 최소 숫자의 인덱스를 찾을 숫자 _리스트_ |

#### 반환 값
최소 숫자의 인덱스

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

숫자들의 평균

| 인수 |  |
| :--- | :--- |
| `numbers` | 평균을 구할 숫자 _리스트_ |

#### 반환 값
숫자들의 평균

<h3 id="numbers-max"><code>numbers-max</code></h3>

최대 숫자

| 인수 |  |
| :--- | :--- |
| `numbers` | 최대 숫자를 찾을 숫자 _리스트_ |

#### 반환 값
최대 숫자

<h3 id="numbers-min"><code>numbers-min</code></h3>

최소 숫자

| 인수 |  |
| :--- | :--- |
| `numbers` | 최소 숫자를 찾을 숫자 _리스트_ |

#### 반환 값
최소 숫자

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

숫자들의 표준편차

| 인수 |  |
| :--- | :--- |
| `numbers` | 표준편차를 계산할 숫자 _리스트_ |

#### 반환 값
숫자들의 표준편차

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

숫자들의 합

| 인수 |  |
| :--- | :--- |
| `numbers` | 합을 구할 숫자 _리스트_ |

#### 반환 값
숫자들의 합

<h3 id="number-toString"><code>number-toString</code></h3>

숫자를 문자열로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `in` | 변환할 숫자 |

#### 반환 값
숫자의 문자열 표현

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

숫자를 _타임스탬프_로 변환합니다. 31536000000보다 작은 값은 초로, 31536000000000보다 작은 값은 밀리초로, 31536000000000000보다 작은 값은 마이크로초로, 31536000000000000000보다 작은 값은 나노초로 변환됩니다.

| 인수 |  |
| :--- | :--- |
| `val` | 타임스탬프로 변환할 숫자 |

#### 반환 값
타임스탬프

<h3 id="number-abs"><code>number-abs</code></h3>

숫자의 절대값을 계산합니다.

| 인수 |  |
| :--- | :--- |
| `n` | 숫자 |

#### 반환 값
숫자의 절대값