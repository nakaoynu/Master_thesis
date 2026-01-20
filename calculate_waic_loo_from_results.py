"""
既存のベイズ推定結果からWAIC/PSIS-LOOを計算するスクリプト
"""
import arviz as az
import numpy as np
import json
import pathlib
import warnings

# 結果ディレクトリ
results_dir = pathlib.Path("bayesian_results_scaled_loocv_20260119_063734")

# トレースファイル読み込み
print("=" * 80)
print("WAIC/PSIS-LOO計算スクリプト")
print("=" * 80)

print(f"\n📂 結果ディレクトリ: {results_dir}")

# H-form
print("\n" + "=" * 80)
print("H形式の計算")
print("=" * 80)

trace_H = az.from_netcdf(results_dir / "trace_H.nc")
print(f"✓ H形式トレース読み込み完了")
print(f"  Groups: {list(trace_H._groups)}")
print(f"  log_likelihood shape: {trace_H.log_likelihood['likelihood'].shape}")

# WAIC計算
print("\n📊 WAIC計算中...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    waic_H = az.waic(trace_H, pointwise=True)

print(f"✓ WAIC計算完了")
print(f"  ELPD WAIC: {waic_H.elpd_waic:.2f} ± {waic_H.se:.2f}")
print(f"  p_waic (有効パラメータ数): {waic_H.p_waic:.2f}")
waic_value_H = -2 * waic_H.elpd_waic  # WAIC = -2 * ELPD_WAIC
print(f"  WAIC: {waic_value_H:.2f}")

# PSIS-LOO計算
print("\n📊 PSIS-LOO計算中...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    loo_H = az.loo(trace_H, pointwise=True)

print(f"✓ PSIS-LOO計算完了")
print(f"  ELPD LOO: {loo_H.elpd_loo:.2f} ± {loo_H.se:.2f}")
print(f"  p_loo (有効パラメータ数): {loo_H.p_loo:.2f}")
loo_value_H = -2 * loo_H.elpd_loo  # LOO = -2 * ELPD_LOO
print(f"  LOO: {loo_value_H:.2f}")

# Pareto k診断
if hasattr(loo_H, 'pareto_k'):
    pareto_k_H = loo_H.pareto_k
    k_good_H = np.sum(pareto_k_H < 0.5)
    k_ok_H = np.sum((pareto_k_H >= 0.5) & (pareto_k_H < 0.7))
    k_bad_H = np.sum((pareto_k_H >= 0.7) & (pareto_k_H < 1.0))
    k_verybad_H = np.sum(pareto_k_H >= 1.0)
    n_total_H = len(pareto_k_H)
    
    print(f"\n  Pareto k診断 (n={n_total_H}):")
    print(f"    k < 0.5 (good): {k_good_H} ({k_good_H/n_total_H*100:.1f}%)")
    print(f"    0.5 ≤ k < 0.7 (ok): {k_ok_H} ({k_ok_H/n_total_H*100:.1f}%)")
    print(f"    0.7 ≤ k < 1.0 (bad): {k_bad_H} ({k_bad_H/n_total_H*100:.1f}%)")
    print(f"    k ≥ 1.0 (very bad): {k_verybad_H} ({k_verybad_H/n_total_H*100:.1f}%)")
    
    if k_verybad_H > 0:
        print(f"  ⚠️ 警告: {k_verybad_H}点でk≥1.0（PSIS-LOOの信頼性が低い）")
    elif k_bad_H > n_total_H * 0.1:
        print(f"  ⚠️ 注意: {k_bad_H}点で0.7≤k<1.0（一部の推定が不安定）")
    else:
        print(f"  ✅ Pareto k値は良好（ほとんどのk<0.7）")

# B-form
print("\n" + "=" * 80)
print("B形式の計算")
print("=" * 80)

trace_B = az.from_netcdf(results_dir / "trace_B.nc")
print(f"✓ B形式トレース読み込み完了")
print(f"  Groups: {list(trace_B._groups)}")
print(f"  log_likelihood shape: {trace_B.log_likelihood['likelihood'].shape}")

# WAIC計算
print("\n📊 WAIC計算中...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    waic_B = az.waic(trace_B, pointwise=True)

print(f"✓ WAIC計算完了")
print(f"  ELPD WAIC: {waic_B.elpd_waic:.2f} ± {waic_B.se:.2f}")
print(f"  p_waic (有効パラメータ数): {waic_B.p_waic:.2f}")
waic_value_B = -2 * waic_B.elpd_waic
print(f"  WAIC: {waic_value_B:.2f}")

# PSIS-LOO計算
print("\n📊 PSIS-LOO計算中...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    loo_B = az.loo(trace_B, pointwise=True)

print(f"✓ PSIS-LOO計算完了")
print(f"  ELPD LOO: {loo_B.elpd_loo:.2f} ± {loo_B.se:.2f}")
print(f"  p_loo (有効パラメータ数): {loo_B.p_loo:.2f}")
loo_value_B = -2 * loo_B.elpd_loo
print(f"  LOO: {loo_value_B:.2f}")

# Pareto k診断
if hasattr(loo_B, 'pareto_k'):
    pareto_k_B = loo_B.pareto_k
    k_good_B = np.sum(pareto_k_B < 0.5)
    k_ok_B = np.sum((pareto_k_B >= 0.5) & (pareto_k_B < 0.7))
    k_bad_B = np.sum((pareto_k_B >= 0.7) & (pareto_k_B < 1.0))
    k_verybad_B = np.sum(pareto_k_B >= 1.0)
    n_total_B = len(pareto_k_B)
    
    print(f"\n  Pareto k診断 (n={n_total_B}):")
    print(f"    k < 0.5 (good): {k_good_B} ({k_good_B/n_total_B*100:.1f}%)")
    print(f"    0.5 ≤ k < 0.7 (ok): {k_ok_B} ({k_ok_B/n_total_B*100:.1f}%)")
    print(f"    0.7 ≤ k < 1.0 (bad): {k_bad_B} ({k_bad_B/n_total_B*100:.1f}%)")
    print(f"    k ≥ 1.0 (very bad): {k_verybad_B} ({k_verybad_B/n_total_B*100:.1f}%)")
    
    if k_verybad_B > 0:
        print(f"  ⚠️ 警告: {k_verybad_B}点でk≥1.0（PSIS-LOOの信頼性が低い）")
    elif k_bad_B > n_total_B * 0.1:
        print(f"  ⚠️ 注意: {k_bad_B}点で0.7≤k<1.0（一部の推定が不安定）")
    else:
        print(f"  ✅ Pareto k値は良好（ほとんどのk<0.7）")

# モデル比較
print("\n" + "=" * 80)
print("モデル比較")
print("=" * 80)

# WAIC比較
elpd_diff_waic = waic_H.elpd_waic - waic_B.elpd_waic
se_diff_waic = np.sqrt(waic_H.se**2 + waic_B.se**2)

print(f"\n📊 ELPD WAIC差分 (H-form - B-form):")
print(f"  ΔELPD: {elpd_diff_waic:.2f} ± {se_diff_waic:.2f}")

if abs(elpd_diff_waic) < 2 * se_diff_waic:
    waic_winner = "引き分け（有意差なし）"
    print(f"  ➡️ 結論: 有意差なし（|ΔELPD| < 2×SE）")
elif elpd_diff_waic > 0:
    waic_winner = "H-form"
    print(f"  🏆 H形式の方が良い（ELPD差: {elpd_diff_waic:.2f}）")
else:
    waic_winner = "B-form"
    print(f"  🏆 B形式の方が良い（ELPD差: {abs(elpd_diff_waic):.2f}）")

# LOO比較
elpd_diff_loo = loo_H.elpd_loo - loo_B.elpd_loo
se_diff_loo = np.sqrt(loo_H.se**2 + loo_B.se**2)

print(f"\n📊 ELPD PSIS-LOO差分 (H-form - B-form):")
print(f"  ΔELPD: {elpd_diff_loo:.2f} ± {se_diff_loo:.2f}")

if abs(elpd_diff_loo) < 2 * se_diff_loo:
    loo_winner = "引き分け（有意差なし）"
    print(f"  ➡️ 結論: 有意差なし（|ΔELPD| < 2×SE）")
elif elpd_diff_loo > 0:
    loo_winner = "H-form"
    print(f"  🏆 H形式の方が良い（ELPD差: {elpd_diff_loo:.2f}）")
else:
    loo_winner = "B-form"
    print(f"  🏆 B形式の方が良い（ELPD差: {abs(elpd_diff_loo):.2f}）")

# サマリー表
print(f"\n📊 モデル比較サマリー:")
print(f"  {'モデル':<10} {'ELPD WAIC':<15} {'ELPD LOO':<15} {'p_waic':<10} {'p_loo':<10}")
print(f"  {'-'*60}")
print(f"  {'H-form':<10} {waic_H.elpd_waic:<15.2f} {loo_H.elpd_loo:<15.2f} {waic_H.p_waic:<10.2f} {loo_H.p_loo:<10.2f}")
print(f"  {'B-form':<10} {waic_B.elpd_waic:<15.2f} {loo_B.elpd_loo:<15.2f} {waic_B.p_waic:<10.2f} {loo_B.p_loo:<10.2f}")

# 統合評価
if waic_winner == loo_winner:
    final_winner = waic_winner
    confidence = "高（WAICとLOOが一致）"
elif "引き分け" in waic_winner or "引き分け" in loo_winner:
    final_winner = "判定保留（WAICとLOOで結果が分かれる）"
    confidence = "中"
else:
    final_winner = "判定保留（WAICとLOOで結果が分かれる）"
    confidence = "低"

print(f"\n🏆 統合評価:")
print(f"  推奨モデル: {final_winner}")
print(f"  信頼性: {confidence}")

# JSON保存
results = {
    "H_form": {
        "waic": {
            "elpd_waic": float(waic_H.elpd_waic),
            "se": float(waic_H.se),
            "p_waic": float(waic_H.p_waic),
            "waic": float(waic_value_H)
        },
        "loo": {
            "elpd_loo": float(loo_H.elpd_loo),
            "se": float(loo_H.se),
            "p_loo": float(loo_H.p_loo),
            "loo": float(loo_value_H),
            "pareto_k": {
                "good": int(k_good_H),
                "ok": int(k_ok_H),
                "bad": int(k_bad_H),
                "very_bad": int(k_verybad_H)
            }
        }
    },
    "B_form": {
        "waic": {
            "elpd_waic": float(waic_B.elpd_waic),
            "se": float(waic_B.se),
            "p_waic": float(waic_B.p_waic),
            "waic": float(waic_value_B)
        },
        "loo": {
            "elpd_loo": float(loo_B.elpd_loo),
            "se": float(loo_B.se),
            "p_loo": float(loo_B.p_loo),
            "loo": float(loo_value_B),
            "pareto_k": {
                "good": int(k_good_B),
                "ok": int(k_ok_B),
                "bad": int(k_bad_B),
                "very_bad": int(k_verybad_B)
            }
        }
    },
    "comparison": {
        "waic": {
            "delta_elpd": float(elpd_diff_waic),
            "se_diff": float(se_diff_waic),
            "winner": waic_winner
        },
        "loo": {
            "delta_elpd": float(elpd_diff_loo),
            "se_diff": float(se_diff_loo),
            "winner": loo_winner
        },
        "final_winner": final_winner,
        "confidence": confidence
    }
}

output_file = results_dir / "waic_loo_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ 結果を保存しました: {output_file}")
print("=" * 80)
