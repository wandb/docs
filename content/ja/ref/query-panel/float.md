---
title: "`float`\n\n`float` の指定は、スカラー値を小数として wandb に記録するために使用されます。このシンプルな型は、実数を wandb\
  \ のデータスペースに格納し、視覚化や追跡に利用する際に便利です。\n\n### 使用例\n\n```python\nimport wandb\n\n# wandb\
  \ の初期化\nrun = wandb.init()\n\n# スカラー値を float 型でログに記録\nrun.log({\"accuracy\": float(0.93)})\n\
  \n# 実行の終了\nrun.finish()\n```\n\n### パラメータ\n\n- **value**: `float`  \n  ログに記録する数値。浮動小数点数を指定します。"
menu:
  reference:
    identifier: ja-ref-query-panel-float
---

## Chainable Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値の不等性を決定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [number](number.md) を別のもので割り、余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る [number](number.md) |
| `rhs` | 割る相手の [number](number.md) |

#### 戻り値
2つの [number](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [number](number.md) を掛ける

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2番目の [number](number.md) |

#### 戻り値
2つの [number](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

ある [number](number.md) を指数に上げる

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数の [number](number.md) |
| `rhs` | 指数の [number](number.md) |

#### 戻り値
基数の [number](number.md) がn乗される

<h3 id="number-add"><code>number-add</code></h3>

2つの [number](number.md) を加える

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2番目の [number](number.md) |

#### 戻り値
2つの [number](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある [number](number.md) を別のものから引く

| 引数 |  |
| :--- | :--- |
| `lhs` | 引く対象の [number](number.md) |
| `rhs` | 引く [number](number.md) |

#### 戻り値
2つの [number](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

ある [number](number.md) を別のもので割る

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る [number](number.md) |
| `rhs` | 割る相手の [number](number.md) |

#### 戻り値
2つの [number](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

ある [number](number.md) が別のものより少ないかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2番目より少ないか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある [number](number.md) が別のものより小さいか等しいかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2番目より小さいか等しいか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値の等価性を決定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある [number](number.md) が別のものより大きいかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2番目より大きいか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある [number](number.md) が別のものより大きいか等しいかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2番目より大きいか等しいか

<h3 id="number-negate"><code>number-negate</code></h3>

ある [number](number.md) を否定する

| 引数 |  |
| :--- | :--- |
| `val` | 否定する [number](number.md) |

#### 戻り値
否定された [number](number.md)

<h3 id="number-toString"><code>number-toString</code></h3>

ある [number](number.md) を文字列に変換する

| 引数 |  |
| :--- | :--- |
| `in` | 変換する [number](number.md) |

#### 戻り値
その [number](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

ある [number](number.md) を _timestamp_ に変換します。31536000000 未満の値は秒に変換され、31536000000000 未満の値はミリ秒に、31536000000000000 未満の値はマイクロ秒に、31536000000000000000 未満の値はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する [number](number.md) |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

ある [number](number.md) の絶対値を計算する

| 引数 |  |
| :--- | :--- |
| `n` | ある [number](number.md) |

#### 戻り値
その [number](number.md) の絶対値

## List Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値の不等性を決定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [number](number.md) を別のもので割り、余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る [number](number.md) |
| `rhs` | 割る相手の [number](number.md) |

#### 戻り値
2つの [number](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [number](number.md) を掛ける

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2番目の [number](number.md) |

#### 戻り値
2つの [number](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

ある [number](number.md) を指数に上げる

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数の [number](number.md) |
| `rhs` | 指数の [number](number.md) |

#### 戻り値
基数の [number](number.md) がn乗される

<h3 id="number-add"><code>number-add</code></h3>

2つの [number](number.md) を加える

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2番目の [number](number.md) |

#### 戻り値
2つの [number](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある [number](number.md) を別のものから引く

| 引数 |  |
| :--- | :--- |
| `lhs` | 引く対象の [number](number.md) |
| `rhs` | 引く [number](number.md) |

#### 戻り値
2つの [number](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

ある [number](number.md) を別のもので割る

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る [number](number.md) |
| `rhs` | 割る相手の [number](number.md) |

#### 戻り値
2つの [number](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

ある [number](number.md) が別のものより少ないかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2番目より少ないか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある [number](number.md) が別のものより小さいか等しいかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2番目より小さいか等しいか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値の等価性を決定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある [number](number.md) が別のものより大きいかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2番目より大きいか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある [number](number.md) が別のものより大きいか等しいかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2番目より大きいか等しいか

<h3 id="number-negate"><code>number-negate</code></h3>

ある [number](number.md) を否定する

| 引数 |  |
| :--- | :--- |
| `val` | 否定する [number](number.md) |

#### 戻り値
否定された [number](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最も大きい [number](number.md) のインデックスを見つける

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [number](number.md) のインデックスを見つけるための _list_ of [numbers](number.md) |

#### 戻り値
最大の [number](number.md) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最も小さい [number](number.md) のインデックスを見つける

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [number](number.md) のインデックスを見つけるための _list_ of [numbers](number.md) |

#### 戻り値
最小の [number](number.md) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[numbers](number.md) の平均

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均を計算する [numbers](number.md) の _list_ |

#### 戻り値
[numbers](number.md) の平均

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大の数値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [number](number.md) を見つける _list_ of [numbers](number.md) |

#### 戻り値
最大の [number](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小の数値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [number](number.md) を見つける _list_ of [numbers](number.md) |

#### 戻り値
最小の [number](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[numbers](number.md) の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算する [numbers](number.md) の _list_ |

#### 戻り値
[numbers](number.md) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[numbers](number.md) の合計

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計を計算する [numbers](number.md) の _list_ |

#### 戻り値
[numbers](number.md) の合計

<h3 id="number-toString"><code>number-toString</code></h3>

ある [number](number.md) を文字列に変換する

| 引数 |  |
| :--- | :--- |
| `in` | 変換する [number](number.md) |

#### 戻り値
その [number](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

ある [number](number.md) を _timestamp_ に変換します。31536000000 未満の値は秒に変換され、31536000000000 未満の値はミリ秒に、31536000000000000 未満の値はマイクロ秒に、31536000000000000000 未満の値はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する [number](number.md) |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

ある [number](number.md) の絶対値を計算する

| 引数 |  |
| :--- | :--- |
| `n` | ある [number](number.md) |

#### 戻り値
その [number](number.md) の絶対値