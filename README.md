# Activation Function Comparison (NumPy)

オリジナル活性化関数と既存関数（`relu` / `tanh` / `sigmoid`）を同条件で比較し、
訓練データ・テストデータの損失曲線を1枚に出力します。

## 実行

```bash
python3 compare_activations.py
```

出力:
- 損失グラフ: `results/loss_comparison.png`
- 端末サマリ: 最終損失・最終精度・指定精度に達したエポック（平均±標準偏差）
- （任意）`n` ごとの loss 減少カーブ: `results/custom_n_comparison.png`

## 主要オプション

```bash
python3 compare_activations.py --epochs 700 --lr 0.06 --hidden 48 --acc-threshold 0.92
```

複数 seed で平均比較:

```bash
python3 compare_activations.py --num-seeds 10 --seed 42
```

特定 seed を明示:

```bash
python3 compare_activations.py --seed-list 7,42,123,999
```

`n` を増やして custom 活性化の比較グラフを作る:

```bash
python3 compare_activations.py --num-seeds 10 --seed 42 --custom-n-list 0,1,2,3,4,5
```

出力先を変更:

```bash
python3 compare_activations.py --custom-n-list 0,2,4,6 --custom-n-plot-out results/n_sweep.png
```

## オリジナル活性化関数の編集場所

- 関数本体: `compare_activations.py` の `custom_activation`
- 導関数: `compare_activations.py` の `custom_activation_grad`

この2つをセットで変更してください。
導関数が間違っていると、学習速度比較が正しく評価できません。

現在の `custom` は、調和振動子の固有関数 `psi_n(x)` を使って
`x + alpha * psi_n(beta * x)` として定義しています。

切り替えパラメータ:

```bash
python3 compare_activations.py --custom-n 2 --custom-alpha 0.75 --custom-beta 1.0
```

## 調和振動子の関数系（LaTeX 記法）

この実装は、無次元化した 1 次元調和振動子の固有関数を使っています。

Hermite 多項式（物理学者の定義）:

```math
\begin{aligned}
H_0(x) &= 1, \\
H_1(x) &= 2x, \\
H_{n+1}(x) &= 2x\,H_n(x) - 2n\,H_{n-1}(x).
\end{aligned}
```

固有関数:

```math
\psi_n(x)=\frac{1}{\sqrt{2^n n! \sqrt{\pi}}}\,H_n(x)\,e^{-x^2/2}
```

導関数:

```math
\frac{d\psi_n(x)}{dx}
=\frac{1}{\sqrt{2^n n! \sqrt{\pi}}}
\left(2n\,H_{n-1}(x)-x\,H_n(x)\right)e^{-x^2/2}
```

活性化関数と勾配:

```math
f(x)=x+\alpha\,\psi_n(\beta x)
```

```math
f'(x)=1+\alpha\beta\,\frac{d\psi_n(\beta x)}{d(\beta x)}
```

実装との対応:
- `compare_activations.py` の `hermite_polynomial`
- `compare_activations.py` の `harmonic_oscillator_eigenfunction`
- `compare_activations.py` の `harmonic_oscillator_eigenfunction_grad`
- `compare_activations.py` の `custom_activation`
- `compare_activations.py` の `custom_activation_grad`
