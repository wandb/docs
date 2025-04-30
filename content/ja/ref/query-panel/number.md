---
title: 数字
menu:
  reference:
    identifier: ja-ref-query-panel-number
---

## Chainable Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

1つの [number](number.md) を別のもので割り、余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[number](number.md) |
| `rhs` | 割るための[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの[numbers](number.md) を掛けます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[number](number.md) |
| `rhs` | 2番目の[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

1つの[number](number.md)を指数に上げます

| 引数 |  |
| :--- | :--- |
| `lhs` | ベース[number](number.md) |
| `rhs` | 指数[number](number.md) |

#### 戻り値
ベースの[numbers](number.md)がn乗されます

<h3 id="number-add"><code>number-add</code></h3>

2つの[numbers](number.md)を加えます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[number](number.md) |
| `rhs` | 2番目の[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の合計

<h3 id="number-sub"><code>number-sub</code></h3>

1つの[number](number.md)を別のものから引きます

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる[number](number.md) |
| `rhs` | 引くための[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の差

<h3 id="number-div"><code>number-div</code></h3>

1つの[number](number.md)を別のもので割ります

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[number](number.md) |
| `rhs` | 割るための[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の商

<h3 id="number-less"><code>number-less</code></h3>

1つの[number](number.md)が別のものより小さいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[number](number.md) |
| `rhs` | 比較対象の[number](number.md) |

#### 戻り値
最初の[number](number.md)が2番目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

1つの[number](number.md)が別のものと等しいかまたは小さいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[number](number.md) |
| `rhs` | 比較対象の[number](number.md) |

#### 戻り値
最初の[number](number.md)が2番目と等しいかまたは小さいかどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

1つの[number](number.md)が別のものより大きいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[number](number.md) |
| `rhs` | 比較対象の[number](number.md) |

#### 戻り値
最初の[number](number.md)が2番目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

1つの[number](number.md)が別のものと等しいかまたは大きいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[number](number.md) |
| `rhs` | 比較対象の[number](number.md) |

#### 戻り値
最初の[number](number.md)が2番目と等しいかまたは大きいかどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[number](number.md)を否定します

| 引数 |  |
| :--- | :--- |
| `val` | 否定する番号 |

#### 戻り値
[number](number.md)

<h3 id="number-toString"><code>number-toString</code></h3>

[number](number.md)を文字列に変換します

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数 |

#### 戻り値
[number](number.md)の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[number](number.md)を _timestamp_ に変換します。31536000000未満の値は秒に、31536000000000未満の値はミリ秒に、31536000000000000未満の値はマイクロ秒に、31536000000000000000未満の値はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[number](number.md)の絶対値を計算します

| 引数 |  |
| :--- | :--- |
| `n` | A [number](number.md) |

#### 戻り値
その[number](number.md)の絶対値


## List Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

1つの[number](number.md)を別のもので割り、余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[number](number.md) |
| `rhs` | 割るための[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの[numbers](number.md)を掛けます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[number](number.md) |
| `rhs` | 2番目の[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

1つの[number](number.md)を指数に上げます

| 引数 |  |
| :--- | :--- |
| `lhs` | ベース[number](number.md) |
| `rhs` | 指数[number](number.md) |

#### 戻り値
ベースの[numbers](number.md)がn乗されます

<h3 id="number-add"><code>number-add</code></h3>

2つの[numbers](number.md)を加えます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[number](number.md) |
| `rhs` | 2番目の[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の合計

<h3 id="number-sub"><code>number-sub</code></h3>

1つの[number](number.md)を別のものから引きます

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる[number](number.md) |
| `rhs` | 引くための[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の差

<h3 id="number-div"><code>number-div</code></h3>

1つの[number](number.md)を別のもので割ります

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[number](number.md) |
| `rhs` | 割るための[number](number.md) |

#### 戻り値
2つの[numbers](number.md)の商

<h3 id="number-less"><code>number-less</code></h3>

1つの[number](number.md)が別のものより小さいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[number](number.md) |
| `rhs` | 比較対象の[number](number.md) |

#### 戻り値
最初の[number](number.md)が2番目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

1つの[number](number.md)が別のものと等しいかまたは小さいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[number](number.md) |
| `rhs` | 比較対象の[number](number.md) |

#### 戻り値
最初の[number](number.md)が2番目と等しいかまたは小さいかどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

1つの[number](number.md)が別のものより大きいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[number](number.md) |
| `rhs` | 比較対象の[number](number.md) |

#### 戻り値
最初の[number](number.md)が2番目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

1つの[number](number.md)が別のものと等しいかまたは大きいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[number](number.md) |
| `rhs` | 比較対象の[number](number.md) |

#### 戻り値
最初の[number](number.md)が2番目と等しいかまたは大きいかどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[number](number.md)を否定します

| 引数 |  |
| :--- | :--- |
| `val` | 否定する数 |

#### 戻り値
[number](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大の[number](number.md)のインデックスを見つけます

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の[number](number.md)のインデックスを見つけるための[numbers](number.md)のリスト |

#### 戻り値
最大の[number](number.md)のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小の[number](number.md)のインデックスを見つけます

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の[number](number.md)のインデックスを見つけるための[numbers](number.md)のリスト |

#### 戻り値
最小の[number](number.md)のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[numbers](number.md)の平均

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均を取る[numbers](number.md)のリスト |

#### 戻り値
[numbers](number.md)の平均

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の[number](number.md)を見つけるための[numbers](number.md)のリスト |

#### 戻り値
最大の[number](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の[number](number.md)を見つけるための[numbers](number.md)のリスト |

#### 戻り値
最小の[number](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[numbers](number.md)の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算するための[numbers](number.md)のリスト |

#### 戻り値
[numbers](number.md)の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[numbers](number.md)の合計

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計を求める[numbers](number.md)のリスト |

#### 戻り値
[numbers](number.md)の合計

<h3 id="number-toString"><code>number-toString</code></h3>

[number](number.md)を文字列に変換します

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数 |

#### 戻り値
[number](number.md)の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[number](number.md)を _timestamp_ に変換します。31536000000未満の値は秒に、31536000000000未満の値はミリ秒に、31536000000000000未満の値はマイクロ秒に、31536000000000000000未満の値はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[number](number.md)の絶対値を計算します

| 引数 |  |
| :--- | :--- |
| `n` | A [number](number.md) |

#### 戻り値
その[number](number.md)の絶対値