# Activation Function Comparison (NumPy)

オリジナル活性化関数と既存関数（`relu` / `tanh` / `sigmoid`）を同条件で比較し、
訓練データ・テストデータの損失曲線を1枚に出力します。

## 実行

```bash
python3 compare_activations.py
```

出力:
- 損失グラフ: `results/loss_comparison.png`
- 端末サマリ: 最終損失・最終精度・指定精度に達したエポック

## 主要オプション

```bash
python3 compare_activations.py --epochs 700 --lr 0.06 --hidden 48 --acc-threshold 0.92
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
