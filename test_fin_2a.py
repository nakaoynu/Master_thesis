"""
Bayesian Hierarchical Analysis with SMC Sampling (v7.1)
Non-centered階層γモデル + SMCサンプリング + 物理的制約ベース事前分布

【v7.1更新】2026-01-18
★ Non-centered Parameterizationへ移行（R-hat収束性改善）
  - Centered版: γ_i ~ N(γ_mean, γ_std) → 「漏斗」問題で収束不良
  - Non-centered版: z_i ~ N(0,1), γ_i = exp(log_μ + log_σ * z_i)
  - 期待効果: R-hat 1.135→<1.01, ESS 8396→>25000

【v7更新】2026-01-14
1. 階層的γモデル導入: γ識別不能性を解消
2. SMCサンプリング: 高品質サンプル取得（ESS>400目標）
3. 物理的制約ベース事前分布: v6結果は参考値のみ
4. 外れ値頑健尤度: StudentT分布（ν=4）
5. 重み設定変更: ポラリトン=2.0, 共振器=1.0, その他=0.01（v7.1で0.1→0.01に変更）

【事前分布設定（v7.1 Non-centered版）】
┌─────────┬──────────────┬────────────────────────────────────────┐
│パラメータ│分布型        │設定根拠                                │
├─────────┼──────────────┼────────────────────────────────────────┤
│ g       │TruncNormal   │理論値g≈2.0 (Gd³⁺), σ=0.05             │
│ a       │HalfNormal    │低値優先、上限10に拡張（v6張り付き対応）│
│ B₄      │LogNormal     │正値保証、低値優先、上限50mKに拡張      │
│ B₆      │Normal        │ゼロ中心対称、範囲[-2mK, +2mK]         │
│ ε_bg   │TruncNormal   │v6平均値中心、σ=0.3（情報強化）        │
│log_γ_mu│Normal        │log空間で定義、μ=log(0.074)            │
│log_γ_sd│HalfNormal    │log空間標準偏差、σ=0.5                 │
│γ_raw_i │Normal(0,1)   │Non-centered: 標準正規分布（独立）      │
│ γ_i    │Deterministic │exp(log_μ + log_σ * z_i) (決定論的変換) │
└─────────┴──────────────┴────────────────────────────────────────┘

【機能】
1. H形式とB形式を両方同時に処理
2. SMCサンプリング（Draws=5000, Chains=8）
3. 重み設定: ポラリトン=2.0, 高次共振器=1.0, それ以外=0.01
4. LOO-CV (Leave-One-Out Cross-Validation): モデル評価と比較
5. デバッグモード: DEBUG_MODE=Trueで高速テスト実行
"""

# ========== 設定（v7階層モデル用）==========
SAMPLER_TYPE = 'SMC'              # SMCで開始（最も安定）
USE_HIERARCHICAL_GAMMA = True     # 階層γモデル必須
USE_V6_AS_REFERENCE_ONLY = True   # v6は参考値のみ（中心値には使わない）
LIKELIHOOD_TYPE = 'studentt'      # 外れ値頑健性
NU_STUDENTT = 4                   # 自由度
RANDOM_SEED = 42                   # 乱数シード固定（再現性確保）

# SMC設定（ESS>400達成向け）
SMC_DRAWS = 10000      # 高品質サンプリング
SMC_CHAINS = 16        # 並列度向上
SMC_PARALLEL = True

# 階層γモデル設定（WNLS最適化結果ベース: H/B形式140サンプル分析）
GAMMA_HYPERPRIOR_MU = 0.074    # 中央値（外れ値ロバスト）
GAMMA_HYPERPRIOR_SIGMA = 0.160  # 全体標準偏差×1.5（広い探索範囲）
GAMMA_STD_PRIOR = 0.092         # データセット内ばらつき（個別γの変動許容）

import os
import sys
import json
import time
import pathlib
import datetime
import warnings
warnings.filterwarnings('ignore')

# CPU環境設定
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans',


import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from scipy.signal import find_peaks
from scipy.stats import truncnorm

# PyTensorの警告を抑制
import logging
logging.getLogger('pytensor').setLevel(logging.ERROR)

# 定数定義（unified_weighted_bayesian_fitting_final.py準拠）
kB = 1.380649e-23      # ボルツマン定数 [J/K]
muB = 9.274010e-24     # ボーア磁子 [J/T]
hbar = 1.054571e-34    # プランク定数 [J·s]
c = 299792458          # 光速 [m/s]
mu0 = 4.0 * np.pi * 1e-7  # 真空透磁率 [H/m]
eps0 = 8.854187817e-12    # 真空誘電率 [F/m]

# THz単位系変換定数
THZ_TO_HZ = 1e12
THZ_TO_RAD_S = 2.0 * np.pi * THZ_TO_HZ  # THz → rad/s
RAD_S_TO_THZ = 1.0 / THZ_TO_RAD_S        # rad/s → THz

# スピン密度と試料厚さ（pre_test_v6_shared_gamma.py準拠）
N_SPIN = 1.9386e+28    # スピン密度 [m^-3]
d_fixed = 157.8e-6     # 試料厚さ [m]

# パラメータスケーリング係数（v7階層モデル対応）
# 目標: 最適化空間で全パラメータの幅を50程度に統一
# v7境界拡張版: a=[0.1, 10.0], B₄=[0.01mK, 50mK], B₆=[-2mK, 2mK]
SCALING_FACTORS = {
    'g': 38.0,      # [1.5, 2.8] → [57, 106] (幅49)
    'a': 10.2,      # [0.1, 10.0] → [1.02, 102.0] (v7: 上限10に拡張)
    'B4': 1672.0,   # [1e-5, 5e-2] → [0.017, 83.6] (v7: 上限50mKに拡張)
    'B6': 25000.0,  # [-2e-3, 2e-3] → [-50, 50] (v7: 範囲拡張)
    'eps': 17.0,    # [13.0, 16.0] → [221, 272] (幅51)
    'gamma': 100.0  # [0.005, 0.5] → [0.5, 50.0] (v7: 下限緩和)
}

# 処理対象データ（全10データセット）
TARGET_DATA = [
    {'B': 9.0, 'T': 4.0,  'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '4K'},
    {'B': 9.0, 'T': 10.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '10K'},
    {'B': 9.0, 'T': 20.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '20K'},
    {'B': 9.0, 'T': 30.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '30K'},
    {'B': 4.2, 'T': 1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx', 'sheet': 'Normalized Data', 'col': '4.2T'},
    {'B': 5.0, 'T': 1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx', 'sheet': 'Normalized Data', 'col': '5T'},
    {'B': 6.0, 'T': 1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx', 'sheet': 'Normalized Data', 'col': '6T'},
    {'B': 7.0, 'T': 1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx', 'sheet': 'Normalized Data', 'col': '7T'},
    {'B': 8.0, 'T': 1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx', 'sheet': 'Normalized Data', 'col': '8T'},
    {'B': 9.0, 'T': 1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx', 'sheet': 'Normalized Data', 'col': '9T'},
]

# ============================================================================
# 物理関数群（unified_weighted_bayesian_fitting_final.py準拠）
# ============================================================================

# スピン量子数とガンマパラメータ数
S_VALUE = 3.5
N_TRANSITIONS = 7  # 7-gamma初期状態ベース方式

def get_hamiltonian(B_ext_z, g_factor, B4, B6, s=S_VALUE):
    """ハミルトニアンを計算する（Stevens演算子使用）"""
    n_states = int(2 * s + 1)
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    
    if n_states == 8:  # s = 7/2 (Gd3+)
        # Stevens演算子 O40, O44（正規化：無次元）
        O40 = np.diag([7, -13, -3, 9, 9, -3, -13, 7]) / 60
        X_O44 = np.zeros((8, 8))
        X_O44[3, 7] = X_O44[4, 0] = np.sqrt(35) / 12
        X_O44[2, 6] = X_O44[5, 1] = 5 * np.sqrt(3) / 12
        O44 = (X_O44 + X_O44.T)
        
        # Stevens演算子 O60, O64（正規化：無次元）
        O60 = np.diag([1, -5, 9, -5, -5, 9, -5, 1]) / 1260
        X_O64 = np.zeros((8, 8))
        X_O64[3, 7] = X_O64[4, 0] = 3 * np.sqrt(35) / 60
        X_O64[2, 6] = X_O64[5, 1] = -7 * np.sqrt(3) / 60
        O64 = (X_O64 + X_O64.T)
    else:
        raise ValueError(f"s={s}の結晶場演算子は未実装です")
    
    # 結晶場ハミルトニアン（K単位）
    H_cf = B4 * (O40 + 5 * O44) + B6 * (O60 - 21 * O64)
    
    # ゼーマン項（Joule単位からK単位に変換）
    H_zee_J = g_factor * muB * B_ext_z * Sz  # Joule
    H_zee = H_zee_J / kB  # K
    
    return H_cf + H_zee


def construct_spin_operators():
    """S=7/2系のスピン演算子 Sx, Sy, Sz の行列を構築"""
    s_val = 3.5
    n_states = int(2 * s_val + 1)  # 8準位
    m_values = np.arange(s_val, -s_val - 1, -1)  # [7/2, 5/2, ..., -7/2]
    
    # Sz は対角行列
    Sz = np.diag(m_values)
    
    # S+ と S- の行列要素を計算
    Sx = np.zeros((n_states, n_states), dtype=float)
    Sy = np.zeros((n_states, n_states), dtype=float)
    
    # S+ の行列要素（対角の1つ上）
    for i in range(n_states - 1):
        m_lower = m_values[i + 1]
        coeff = np.sqrt((s_val - m_lower) * (s_val + m_lower + 1))
        Sx[i, i + 1] += coeff / 2.0
        Sy[i, i + 1] += -coeff / (2.0j)
    
    # S- の行列要素（対角の1つ下）
    for i in range(1, n_states):
        m_upper = m_values[i - 1]
        coeff = np.sqrt((s_val + m_upper) * (s_val - m_upper + 1))
        Sx[i, i - 1] += coeff / 2.0
        Sy[i, i - 1] += coeff / (2.0j)
    
    # Syは複素数行列だが、虚部を実数として扱う（Sy = -i(S+ - S-)/2 の実部）
    # 物理的に正しい実数Sy行列を返す
    Sy_real = np.imag(Sy)  # 虚部が実際のSy成分
    return Sx, Sy_real, Sz


def calculate_susceptibility(freq_thz, H, T, gamma_thz):
    """磁気感受率を厳密に計算（全56遷移考慮、7-gamma初期状態ベース方式）"""
    # γの処理
    if np.isscalar(gamma_thz):
        gamma_mode = 'uniform'
        gamma_uniform = float(gamma_thz)
        gamma_array_7 = None
    elif hasattr(gamma_thz, '__len__'):
        gamma_array = np.atleast_1d(gamma_thz)
        if len(gamma_array) == 7:
            gamma_mode = '7gamma'
            gamma_array_7 = gamma_array
        else:
            gamma_mode = 'uniform'
            gamma_uniform = float(gamma_array[0])
            gamma_array_7 = None
    else:
        gamma_mode = 'uniform'
        gamma_uniform = float(gamma_thz)
        gamma_array_7 = None
    
    # 1. ハミルトニアンを対角化
    eigenvalues_K, eigenvectors = np.linalg.eigh(H)
    E_min = np.min(eigenvalues_K)
    E_shifted_K = eigenvalues_K - E_min  # [K]
    
    # 2. Boltzmann因子計算
    boltzmann_exp = np.clip(E_shifted_K / T, -700, 700)
    Z = np.sum(np.exp(-boltzmann_exp))
    populations = np.exp(-boltzmann_exp) / Z
    
    # 3. エネルギー準位をJ単位に変換
    E_shifted_J = E_shifted_K * kB  # [J]
    
    # 4. スピン演算子の構築
    Sx_zeeman, Sy_zeeman, Sz_zeeman = construct_spin_operators()
    
    # 5. 結晶場固有状態での遷移行列要素を計算
    Sx_eigenbasis = eigenvectors.T.conj() @ Sx_zeeman @ eigenvectors
    Sy_eigenbasis = eigenvectors.T.conj() @ Sy_zeeman @ eigenvectors
    Sz_eigenbasis = eigenvectors.T.conj() @ Sz_zeeman @ eigenvectors
    
    # 磁気感受率テンソルの対角成分
    transition_xx = np.abs(Sx_eigenbasis)**2
    transition_yy = np.abs(Sy_eigenbasis)**2
    transition_zz = np.abs(Sz_eigenbasis)**2
    
    # 6. 全遷移ペアのエネルギー差を計算
    delta_E_matrix = E_shifted_J[None, :] - E_shifted_J[:, None]  # (8, 8), [J]
    omega_0_rad = delta_E_matrix / hbar  # [rad/s]
    freq_0_matrix = omega_0_rad * RAD_S_TO_THZ  # [THz]
    
    # 7. 実効的な遷移強度（面内平均）
    transition_perp = (transition_xx + transition_yy) / 2.0
    
    # 占有確率差分
    pop_diff_matrix = populations[:, None] - populations[None, :]
    
    # Boltzmann重み付き遷移強度
    strength_matrix = pop_diff_matrix * transition_perp
    
    # 8. 対角要素を除外
    non_diag_mask = ~np.eye(8, dtype=bool)
    
    # 低温での分裂構造保持
    population_threshold = 1e-3
    occupied_mask = populations[:, None] > population_threshold
    
    # 有限値チェック
    finite_mask = (
        np.isfinite(freq_0_matrix) & 
        np.isfinite(strength_matrix) & 
        (np.abs(strength_matrix) > 1e-20) &
        occupied_mask &
        non_diag_mask
    )
    
    if not np.any(finite_mask):
        return np.zeros_like(freq_thz, dtype=complex)
    
    # 9. 有効な遷移のみ抽出
    freq_0_valid = freq_0_matrix[finite_mask]
    strength_valid = strength_matrix[finite_mask]
    
    # γ値の割り当て（遷移ごと）
    n_indices, n_prime_indices = np.where(finite_mask)
    
    # エネルギー準位でソート
    energy_order = np.argsort(E_shifted_J)
    
    if gamma_mode == 'uniform':
        gamma_per_transition = np.full(len(freq_0_valid), gamma_uniform)
    elif gamma_mode == '7gamma':
        # 7-gamma: 初期状態ベース方式
        gamma_per_transition = np.zeros(len(freq_0_valid))
        
        for trans_idx in range(len(freq_0_valid)):
            n = n_indices[trans_idx]
            n_prime = n_prime_indices[trans_idx]
            
            # エネルギーの低い方の準位を取得
            E_n = E_shifted_J[n]
            E_n_prime = E_shifted_J[n_prime]
            
            if E_n <= E_n_prime:
                lower_state = n
            else:
                lower_state = n_prime
            
            # エネルギー順でのインデックスを取得
            lower_state_energy_idx = np.where(energy_order == lower_state)[0][0]
            
            # 対応するγを選択（準位0〜6に対応するγ_0〜γ_6）
            gamma_idx = min(lower_state_energy_idx, 6)
            gamma_per_transition[trans_idx] = gamma_array_7[gamma_idx]
    else:
        gamma_per_transition = np.full(len(freq_0_valid), 0.1)
    
    # 10. 感受率計算
    freq_diff = freq_0_valid[None, :] - freq_thz[:, None]
    denominator = freq_diff - 1j * gamma_per_transition[None, :]
    
    # ゼロ除算回避
    safe_mask = np.abs(denominator) > 1e-10
    denominator = np.where(safe_mask, denominator, 1e-10 + 1j * 1e-10)
    
    # 各周波数に対する全遷移の寄与を合計
    chi_array = np.sum(strength_valid[None, :] / denominator, axis=1)
    
    return chi_array


def calculate_transmission(freq_thz, mu_r, d, eps_bg):
    """透過率を計算する（Fabry-Perot干渉考慮）"""
    eps_bg = max(eps_bg, 0.1)
    d = max(d, 1e-6)
    
    omega = freq_thz * THZ_TO_RAD_S
    
    # 比透磁率の安全処理
    mu_r_safe = np.where(np.isfinite(mu_r), mu_r, 1.0 + 0j)
    eps_mu = eps_bg * mu_r_safe
    eps_mu = np.where(eps_mu.real > 0, eps_mu, 0.1 + 1j * eps_mu.imag)
    
    # 複素屈折率と複素インピーダンス
    n_complex = np.sqrt(eps_mu + 0j)
    impe = np.sqrt(mu_r_safe / eps_bg + 0j)
    
    # 波長と位相
    lambda_0 = np.where(omega > 1e-12, (2 * np.pi * c) / omega, np.inf)
    delta = 2 * np.pi * n_complex * d / lambda_0
    delta = np.clip(delta.real, -700, 700) + 1j * np.clip(delta.imag, -700, 700)
    
    # Fabry-Perot透過率
    numerator = 4 * impe
    exp_pos = np.exp(-1j * delta)
    exp_neg = np.exp(1j * delta)
    denominator = (1 + impe)**2 * exp_pos - (1 - impe)**2 * exp_neg
    
    safe_mask = np.abs(denominator) > 1e-15
    t = np.zeros_like(denominator, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    
    transmission = np.abs(t)**2
    transmission = np.where(np.isfinite(transmission), transmission, 0.0)
    transmission = np.clip(transmission, 0, 2)
    
    # Min-Max正規化
    t_min, t_max = np.min(transmission), np.max(transmission)
    if t_max > t_min and np.isfinite(t_max) and np.isfinite(t_min):
        normalized = (transmission - t_min) / (t_max - t_min)
        return np.clip(normalized, 0.0, 1.0)
    else:
        return np.full_like(transmission, 0.5)


# ============================================================================
# v6結果読み込み
# ============================================================================
def load_v6_optimized_params(model_form='H'):
    """v6最適化結果の読み込み（pre_test_v6_shared_gamma.py形式）"""
    json_path = pathlib.Path(__file__).parent / f"global_fitting_results_{model_form}_v6" / "shared_gamma_params.json"
    
    if not json_path.exists():
        print(f"❌ {json_path} が見つかりません")
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # pre_test_v6のJSON構造に対応
        global_params = data['global_params']
        
        params = {
            'g': global_params['g'],
            'a': global_params['a'],
            'B4': global_params['B4'],
            'B6': global_params['B6'],
            'eps': global_params['eps'],
            'gamma': np.array(data['shared_gamma']),
            'cost': data.get('final_cost', None),
            'condition_number': data.get('condition_number', None)
        }
        
        print(f"\n✓ {model_form}-form v6最適化結果:")
        print(f"  g = {params['g']:.6f}")
        print(f"  a = {params['a']:.6f}")
        print(f"  B4 = {params['B4']:.8f}")
        print(f"  B6 = {params['B6']:.8f}")
        print(f"  eps = {params['eps']:.6f}")
        print(f"  gamma = {params['gamma']}")
        if params['cost']:
            print(f"  cost = {params['cost']:.1f}")
        if params['condition_number']:
            print(f"  κ = {params['condition_number']:.2e}")
        
        return params
        
    except Exception as e:
        print(f"❌ {model_form}-form読み込みエラー: {e}")
        return None


# ============================================================================
# データ読み込み
# ============================================================================
def detect_peaks_and_classify(freq, trans, polariton_upper=0.361505, cavity_lower=0.45):
    """ピーク検出とポラリトン/共振器分類（透過スペクトルのピーク検出）"""
    # 透過率の極大値を検出（pre_test_v6と同期）
    peaks, properties = find_peaks(trans, prominence=0.05, width=3)
    
    if len(peaks) == 0:
        return [], []
    
    peak_freqs = freq[peaks]
    peak_widths = properties['widths'] * (freq[1] - freq[0])
    
    sort_idx = np.argsort(peak_freqs)
    peak_freqs = peak_freqs[sort_idx]
    peak_widths = peak_widths[sort_idx]
    
    polariton_regions = []
    cavity_regions = []
    
    for pf, pw in zip(peak_freqs, peak_widths):
        f_start = max(freq[0], pf - 1.5 * pw)
        f_end = min(freq[-1], pf + 1.5 * pw)
        
        if pf <= polariton_upper:
            f_end_clipped = min(f_end, polariton_upper)
            if f_end_clipped > f_start:
                polariton_regions.append((f_start, f_end_clipped))
        elif pf >= cavity_lower:
            f_start_clipped = max(f_start, cavity_lower)
            if f_end > f_start_clipped:
                cavity_regions.append((f_start_clipped, f_end))
    
    return polariton_regions, cavity_regions


def create_weight_array(freq, trans, polariton_regions, cavity_regions):
    """重み配列生成: ポラリトン=2.0, 共振器=1.0, それ以外=0.01（v7更新）"""
    weight_array = np.full_like(freq, 0.01)  
    
    # ポラリトン領域: 2.0（1.5から変更）
    for f_start, f_end in polariton_regions:
        mask = (freq >= f_start) & (freq <= f_end)
        weight_array[mask] = 2.0
    
    # 共振器領域: 1.0
    for f_start, f_end in cavity_regions:
        mask = (freq >= f_start) & (freq <= f_end)
        weight_array[mask] = 1.0
    
    return weight_array


def load_all_datasets(target_data_list):
    """複数データセット読み込み（pre_test_v6互換）"""
    print("\n--- データ読み込み ---")
    
    datasets = []
    base_dir = pathlib.Path(__file__).parent / 'bayesian_inputs'
    
    for idx, config in enumerate(target_data_list):
        excel_path = base_dir / config['file']
        
        if not excel_path.exists():
            print(f"❌ {excel_path} が見つかりません")
            continue
        
        try:
            # Excelシートと列名から読み込み
            df = pd.read_excel(excel_path, sheet_name=config['sheet'])
            
            # 周波数列（共通）
            if 'Frequency (THz)' not in df.columns:
                print(f"❌ {config['col']}: 周波数列が見つかりません")
                continue
            
            # データ列（各条件）
            if config['col'] not in df.columns:
                print(f"❌ {config['col']}: データ列が見つかりません")
                continue
            
            df_clean = df[['Frequency (THz)', config['col']]].dropna()
            freq = df_clean['Frequency (THz)'].values.astype(np.float64)
            trans = df_clean[config['col']].values.astype(np.float64)
            
            # 周波数ごとの重み配列生成
            polariton_regions, cavity_regions = detect_peaks_and_classify(freq, trans)
            weight_array = create_weight_array(freq, trans, polariton_regions, cavity_regions)
            
            # ラベル生成（B/T条件）
            if config['T'] == 1.5:
                label = f"{config['B']:.1f}T"
            else:
                label = f"{config['T']:.0f}K"
            
            dataset = {
                'freq': freq,
                'trans': trans,
                'weight': weight_array,
                'B': config['B'],
                'T': config['T'],
                'label': label,
                'polariton_regions': polariton_regions,
                'cavity_regions': cavity_regions,
                'sigma': np.full_like(freq, 0.01)  # 均一なノイズレベル
            }
            
            datasets.append(dataset)
            
            print(f"✓ {label} (B={config['B']}T, T={config['T']}K): {len(freq)} points")
            print(f"  Polariton領域 (2.0×): {len(polariton_regions)} regions")
            print(f"  Cavity領域 (1.0×): {len(cavity_regions)} regions")
            
        except Exception as e:
            print(f"❌ {config['col']} 読み込みエラー: {e}")
            continue
    
    if len(datasets) == 0:
        print("❌ データセットが1つも読み込めませんでした")
    else:
        print(f"\n✅ 合計 {len(datasets)} データセット読み込み完了")
    
    return datasets


# ============================================================================
# PyTensor Op (スケーリング対応版)
# ============================================================================
class ScaledInformedPriorModelOp(Op):
    """スケーリング対応の情報的事前分布モデルOp（H/B形式選択可能）"""
    
    def __init__(self, datasets, model_form='H'):
        self.datasets = datasets
        self.model_form = model_form
    
    def make_node(self, a_scale_scaled, gamma_vec_scaled, g_factor_scaled, 
                  B4_scaled, B6_scaled, eps_bg_scaled):
        a_scale_scaled = pt.as_tensor_variable(a_scale_scaled)
        gamma_vec_scaled = pt.as_tensor_variable(gamma_vec_scaled)
        g_factor_scaled = pt.as_tensor_variable(g_factor_scaled)
        B4_scaled = pt.as_tensor_variable(B4_scaled)
        B6_scaled = pt.as_tensor_variable(B6_scaled)
        eps_bg_scaled = pt.as_tensor_variable(eps_bg_scaled)
        
        n_total = sum(len(d['freq']) for d in self.datasets)
        output = pt.dvector()
        
        return Apply(self, 
                    [a_scale_scaled, gamma_vec_scaled, g_factor_scaled, 
                     B4_scaled, B6_scaled, eps_bg_scaled],
                    [output])
    
    def perform(self, node, inputs, output_storage):
        a_scale_scaled, gamma_vec_scaled, g_factor_scaled, B4_scaled, B6_scaled, eps_bg_scaled = inputs
        
        # スケーリングされたパラメータを物理値に変換
        g_factor = g_factor_scaled / SCALING_FACTORS['g']
        a_scale = a_scale_scaled / SCALING_FACTORS['a']
        B4 = B4_scaled / SCALING_FACTORS['B4']
        B6 = B6_scaled / SCALING_FACTORS['B6']
        eps_bg = eps_bg_scaled / SCALING_FACTORS['eps']
        
        gamma_array_scaled = np.atleast_1d(gamma_vec_scaled).astype(np.float64)
        if len(gamma_array_scaled) != 7:
            gamma_array_scaled = np.full(7, gamma_array_scaled[0])
        gamma_array = gamma_array_scaled / SCALING_FACTORS['gamma']
        
        all_trans_pred = []
        
        for data in self.datasets:
            freq = data['freq']
            B = data['B']
            T = data['T']
            
            H_ham = get_hamiltonian(B, g_factor, B4, B6)
            chi_raw = calculate_susceptibility(freq, H_ham, T, gamma_array)
            
            G0 = a_scale * mu0 * N_SPIN * (g_factor * muB)**2 / (2 * hbar) / THZ_TO_RAD_S
            chi = G0 * chi_raw
            
            if self.model_form == 'H':
                # H形式: μr = 1 + χ
                mu_r = 1.0 + chi
            else:
                # B形式: μr = 1 / (1 - χ)
                denominator = 1.0 - chi
                mu_r = 1.0 / denominator
            
            trans_pred = calculate_transmission(freq, mu_r, d_fixed, eps_bg)
            
            all_trans_pred.append(trans_pred)
        
        output_storage[0][0] = np.concatenate(all_trans_pred)


# ============================================================================
# モデル評価関数（SMC対応: WAICベース）
# ============================================================================
def compute_model_evaluation(trace, model_name='Model'):
    """
    モデル評価（SMCサンプリング対応）
    
    SMCはlog_likelihoodを自動保存しないため、WAICを使用。
    WAICが計算できない場合は事後予測サマリーを返す。
    """
    print(f"\n{'='*80}")
    print(f"モデル評価: {model_name}")
    print(f"{'='*80}")
    
    result = {'model_name': model_name}
    
    # 1. 基本統計量
    try:
        summary = az.summary(trace)
        n_params = len(summary)
        mean_rhat = summary['r_hat'].mean() if 'r_hat' in summary.columns else np.nan
        mean_ess = summary['ess_bulk'].mean() if 'ess_bulk' in summary.columns else np.nan
        
        print(f"\n📊 {model_name} サンプリング統計:")
        print(f"  パラメータ数: {n_params}")
        if not np.isnan(mean_rhat):
            print(f"  平均R-hat: {mean_rhat:.4f}")
        if not np.isnan(mean_ess):
            print(f"  平均ESS (bulk): {mean_ess:.1f}")
        
        result['n_params'] = n_params
        result['mean_rhat'] = float(mean_rhat) if not np.isnan(mean_rhat) else None
        result['mean_ess'] = float(mean_ess) if not np.isnan(mean_ess) else None
        
    except Exception as e:
        print(f"  ⚠️ サマリー計算エラー: {e}")
    
    # 2. WAIC計算を試みる（log_likelihoodが必要）
    try:
        if 'log_likelihood' in trace:
            waic = az.waic(trace, pointwise=True)
            print(f"\n📊 {model_name} WAIC統計:")
            print(f"  ELPD WAIC: {waic.elpd_waic:.2f} ± {waic.se:.2f}")
            print(f"  p_waic (有効パラメータ数): {waic.p_waic:.2f}")
            
            result['elpd_waic'] = float(waic.elpd_waic)
            result['se_waic'] = float(waic.se)
            result['p_waic'] = float(waic.p_waic)
            result['has_waic'] = True
        else:
            print(f"\n  ⚠️ log_likelihoodが保存されていないためWAIC計算をスキップ")
            print(f"     (SMCサンプリングではlog_likelihoodが自動保存されません)")
            result['has_waic'] = False
            
    except Exception as e:
        print(f"  ⚠️ WAIC計算エラー: {e}")
        result['has_waic'] = False
    
    # 3. 事後予測の基本統計
    try:
        posterior = trace.posterior
        n_chains = posterior.dims.get('chain', 1)
        n_draws = posterior.dims.get('draw', 0)
        total_samples = n_chains * n_draws
        
        print(f"\n📊 {model_name} 事後分布統計:")
        print(f"  チェーン数: {n_chains}")
        print(f"  サンプル数/チェーン: {n_draws}")
        print(f"  総サンプル数: {total_samples}")
        
        result['n_chains'] = n_chains
        result['n_draws'] = n_draws
        result['total_samples'] = total_samples
        
    except Exception as e:
        print(f"  ⚠️ 事後分布統計エラー: {e}")
    
    return result


def compare_models(eval_H, eval_B):
    """H形式とB形式のモデル比較"""
    print(f"\n{'='*80}")
    print(f"モデル比較")
    print(f"{'='*80}")
    
    try:
        # WAIC比較（利用可能な場合）
        if eval_H.get('has_waic') and eval_B.get('has_waic'):
            elpd_H = eval_H['elpd_waic']
            elpd_B = eval_B['elpd_waic']
            se_H = eval_H['se_waic']
            se_B = eval_B['se_waic']
            
            elpd_diff = elpd_H - elpd_B
            se_diff = np.sqrt(se_H**2 + se_B**2)
            
            print(f"\n📊 ELPD WAIC差分 (H-form - B-form):")
            print(f"  ΔELPD: {elpd_diff:.2f} ± {se_diff:.2f}")
            
            if abs(elpd_diff) < 2 * se_diff:
                print(f"  → モデル間に有意な差はありません（|ΔELPD| < 2×SE）")
                winner = "引き分け"
            elif elpd_diff > 0:
                print(f"  → H形式が優れています（ΔELPD > 2×SE）")
                winner = "H-form"
            else:
                print(f"  → B形式が優れています（ΔELPD < -2×SE）")
                winner = "B-form"
            
            # 比較サマリー
            print(f"\n📊 モデル比較サマリー:")
            print(f"  {'モデル':<10} {'ELPD WAIC':<15} {'SE':<10} {'p_waic':<10}")
            print(f"  {'-'*45}")
            print(f"  {'H-form':<10} {elpd_H:<15.2f} {se_H:<10.2f} {eval_H['p_waic']:<10.2f}")
            print(f"  {'B-form':<10} {elpd_B:<15.2f} {se_B:<10.2f} {eval_B['p_waic']:<10.2f}")
            print(f"\n  🏆 推奨モデル: {winner}")
            
            return {
                'elpd_diff': elpd_diff,
                'se_diff': se_diff,
                'winner': winner,
                'method': 'WAIC'
            }
        else:
            # WAICが利用できない場合はESS比較
            print(f"\n  ⚠️ WAICが利用できないため、サンプリング品質で比較")
            
            ess_H = eval_H.get('mean_ess', 0)
            ess_B = eval_B.get('mean_ess', 0)
            
            print(f"\n📊 ESS (Effective Sample Size) 比較:")
            print(f"  H-form ESS: {ess_H:.1f}")
            print(f"  B-form ESS: {ess_B:.1f}")
            
            if ess_H > ess_B * 1.1:
                winner = "H-form (より良いサンプリング)"
            elif ess_B > ess_H * 1.1:
                winner = "B-form (より良いサンプリング)"
            else:
                winner = "引き分け"
            
            print(f"\n  🏆 推奨モデル: {winner}")
            
            return {
                'ess_H': ess_H,
                'ess_B': ess_B,
                'winner': winner,
                'method': 'ESS comparison'
            }
    
    except Exception as e:
        print(f"❌ モデル比較エラー: {e}")
        return None


# 後方互換性のためのエイリアス
def compute_loo_cv(trace, model_name='Model'):
    """後方互換性のためのエイリアス（compute_model_evaluationを呼び出す）"""
    return compute_model_evaluation(trace, model_name)


def compare_models_loo(eval_H, eval_B):
    """後方互換性のためのエイリアス（compare_modelsを呼び出す）"""
    return compare_models(eval_H, eval_B)


# ============================================================================
# ベイズファクター計算（SMC対応）
# ============================================================================
def compute_bayes_factor_smc(trace_H, trace_B):
    """
    SMCサンプリング結果からベイズファクターを推定
    
    SMCサンプラーは周辺尤度（marginal likelihood）の推定値を
    sample_stats.log_marginal_likelihoodとして保存します。
    
    ベイズファクター: BF_{H/B} = P(D|M_H) / P(D|M_B)
    対数ベイズファクター: log(BF) = log(P(D|M_H)) - log(P(D|M_B))
    
    Jeffreys (1961) の解釈基準:
    |log₁₀(BF)|  |ln(BF)|   強さ
    0 - 0.5      0 - 1.15   ほぼ証拠なし
    0.5 - 1      1.15 - 2.3 弱い証拠
    1 - 2        2.3 - 4.6  中程度の証拠
    > 2          > 4.6      強い証拠
    
    Returns:
        dict: ベイズファクター関連の統計量
    """
    print(f"\n{'='*80}")
    print("ベイズファクター計算（SMC周辺尤度ベース）")
    print(f"{'='*80}")
    
    result = {}
    
    try:
        # SMCのsample_statsから周辺尤度を取得
        has_lml_H = hasattr(trace_H, 'sample_stats') and 'log_marginal_likelihood' in trace_H.sample_stats
        has_lml_B = hasattr(trace_B, 'sample_stats') and 'log_marginal_likelihood' in trace_B.sample_stats
        
        if not has_lml_H or not has_lml_B:
            print("  ⚠️ SMC周辺尤度が保存されていません")
            print("     PyMC >= 5.0 の sample_smc() で計算されます")
            
            # 代替: WAIC差分からの近似BF（Bridge Samplingの代替）
            print("\n  → WAICベースの近似ベイズファクターを計算...")
            return _compute_approximate_bf_from_waic(trace_H, trace_B)
        
        # 周辺尤度の取得（全チェーンの平均）
        lml_H = float(trace_H.sample_stats['log_marginal_likelihood'].values.mean())
        lml_B = float(trace_B.sample_stats['log_marginal_likelihood'].values.mean())
        
        # チェーン間の標準偏差（不確実性推定）
        lml_H_std = float(trace_H.sample_stats['log_marginal_likelihood'].values.std())
        lml_B_std = float(trace_B.sample_stats['log_marginal_likelihood'].values.std())
        
        # 対数ベイズファクター（H形式 vs B形式）
        log_BF = lml_H - lml_B
        log_BF_se = np.sqrt(lml_H_std**2 + lml_B_std**2)
        
        # log10スケールへの変換
        log10_BF = log_BF / np.log(10)
        
        print(f"\n📊 周辺尤度 (log scale):")
        print(f"  H形式: {lml_H:.2f} ± {lml_H_std:.2f}")
        print(f"  B形式: {lml_B:.2f} ± {lml_B_std:.2f}")
        print(f"\n📊 ベイズファクター:")
        print(f"  log(BF_{{H/B}}): {log_BF:.2f} ± {log_BF_se:.2f}")
        print(f"  log₁₀(BF_{{H/B}}): {log10_BF:.2f}")
        
        # Jeffreysの解釈基準
        abs_log_BF = abs(log_BF)
        if abs_log_BF < 1.15:  # |log10| < 0.5
            strength = "ほぼ証拠なし (Barely worth mentioning)"
        elif abs_log_BF < 2.3:  # |log10| < 1
            strength = "弱い証拠 (Substantial)"
        elif abs_log_BF < 4.6:  # |log10| < 2
            strength = "中程度の証拠 (Strong)"
        else:
            strength = "強い証拠 (Decisive)"
        
        if log_BF > 0:
            winner = "H-form"
            favor = "H形式を支持"
        elif log_BF < 0:
            winner = "B-form"
            favor = "B形式を支持"
        else:
            winner = "引き分け"
            favor = "どちらも同等"
        
        print(f"\n📊 Jeffreys (1961) 解釈:")
        print(f"  {favor}: {strength}")
        print(f"\n  🏆 推奨モデル: {winner}")
        
        result = {
            'log_marginal_likelihood_H': lml_H,
            'log_marginal_likelihood_B': lml_B,
            'log_marginal_likelihood_H_std': lml_H_std,
            'log_marginal_likelihood_B_std': lml_B_std,
            'log_BF': log_BF,
            'log_BF_se': log_BF_se,
            'log10_BF': log10_BF,
            'interpretation': strength,
            'winner': winner,
            'method': 'SMC marginal likelihood'
        }
        
    except Exception as e:
        print(f"❌ ベイズファクター計算エラー: {e}")
        import traceback
        traceback.print_exc()
        result = {'error': str(e), 'winner': 'N/A'}
    
    return result


def _compute_approximate_bf_from_waic(trace_H, trace_B):
    """
    WAICベースの近似ベイズファクター計算
    
    WAIC（広く適用可能な情報量規準）からELPDを用いて
    ベイズファクターを近似する方法。
    
    注意: これは厳密なベイズファクターではなく、
    予測性能に基づく近似値です。
    """
    result = {'method': 'WAIC approximation (fallback)'}
    
    try:
        # WAIC計算
        if 'log_likelihood' not in trace_H or 'log_likelihood' not in trace_B:
            print("  ⚠️ log_likelihoodが保存されていないため計算不可")
            result['error'] = 'log_likelihood not available'
            result['winner'] = 'N/A'
            return result
        
        waic_H = az.waic(trace_H)
        waic_B = az.waic(trace_B)
        
        # ELPD差分からの近似log(BF)
        # ELPD ≈ log(predictive performance) なので
        # ΔELPD ≈ log(BF) の近似として使用
        elpd_diff = waic_H.elpd_waic - waic_B.elpd_waic
        se_diff = np.sqrt(waic_H.se**2 + waic_B.se**2)
        
        print(f"\n📊 WAIC近似ベイズファクター:")
        print(f"  ELPD H形式: {waic_H.elpd_waic:.2f} ± {waic_H.se:.2f}")
        print(f"  ELPD B形式: {waic_B.elpd_waic:.2f} ± {waic_B.se:.2f}")
        print(f"  ΔELPD (≈log BF): {elpd_diff:.2f} ± {se_diff:.2f}")
        
        # 有意性判定
        if abs(elpd_diff) < 2 * se_diff:
            winner = "引き分け"
            interpretation = "有意な差なし"
        elif elpd_diff > 0:
            winner = "H-form"
            interpretation = "H形式が優れた予測性能"
        else:
            winner = "B-form"
            interpretation = "B形式が優れた予測性能"
        
        print(f"\n  {interpretation}")
        print(f"  🏆 推奨モデル: {winner}")
        
        result.update({
            'elpd_H': float(waic_H.elpd_waic),
            'elpd_B': float(waic_B.elpd_waic),
            'elpd_diff': float(elpd_diff),
            'se_diff': float(se_diff),
            'log_BF': float(elpd_diff),  # 近似値として
            'interpretation': interpretation,
            'winner': winner
        })
        
    except Exception as e:
        print(f"  ⚠️ WAIC近似計算エラー: {e}")
        result['error'] = str(e)
        result['winner'] = 'N/A'
    
    return result


# ============================================================================
# プロット関数（既存のものを流用、calculate_transmission_for_paramsのみ追加）
# ============================================================================
def calculate_transmission_for_params(freq, B, T, g, a, B4, B6, eps, gamma_array, model_form='H'):
    """指定パラメータで透過スペクトルを計算"""
    H_ham = get_hamiltonian(B, g, B4, B6)
    chi_raw = calculate_susceptibility(freq, H_ham, T, gamma_array)
    G0 = a * mu0 * N_SPIN * (g * muB)**2 / (2 * hbar) / THZ_TO_RAD_S
    chi = G0 * chi_raw
    
    if model_form == 'H':
        mu_r = 1.0 + chi
    else:
        denominator = 1.0 - chi
        mu_r = 1.0 / denominator
    
    trans = calculate_transmission(freq, mu_r, d_fixed, eps)
    return trans


def plot_prior_distributions(v6_params, model_form='H', save_dir=None):
    """
    事前分布の可視化（物理値空間）- v7階層モデル対応
    
    v7の新しい事前分布:
    - g: TruncatedNormal(μ=2.0, σ=0.05)
    - a: HalfNormal(σ=2.0) + clip[0.1, 10]
    - B₄: LogNormal(μ=log(2mK), σ=1.2) + clip[0.01mK, 50mK]
    - B₆: Normal(μ=0, σ=0.5mK) + clip[-2mK, 2mK]
    - ε_bg: TruncatedNormal(μ=v6平均, σ=0.3)
    - γ_mean: TruncatedNormal(μ=0.07, σ=0.04)
    - γ_std: HalfNormal(σ=0.03)
    - γ_i: TruncatedNormal(μ=γ_mean, σ=γ_std) (階層モデル)
    """
    from scipy.stats import halfnorm, lognorm, norm
    
    print(f"\n{'='*80}")
    print(f"事前分布プロット作成 ({model_form}-form) [v7階層モデル]")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 13))
    fig.suptitle(f'Prior Distributions ({model_form}-form) - v7 Hierarchical', fontsize=16, y=0.98)
    axes = axes.flatten()
    
    # 1. g_factor: TruncatedNormal(μ=2.0, σ=0.05, [1.5, 2.8])
    ax = axes[0]
    g_range = np.linspace(1.5, 2.8, 500)
    a_trunc, b_trunc = (1.5 - 2.0) / 0.05, (2.8 - 2.0) / 0.05
    g_prior = truncnorm.pdf(g_range, a_trunc, b_trunc, loc=2.0, scale=0.05)
    ax.plot(g_range, g_prior, 'b-', lw=2, label='Prior')
    ax.axvline(v6_params['g'], color='r', linestyle='--', lw=1.5, label=f'v6: {v6_params["g"]:.2f}')
    ax.axvline(2.0, color='g', linestyle=':', lw=1.5, label='Theory: 2.0')
    ax.set_xlabel('g-factor', fontsize=9)
    ax.set_ylabel('Prob. Density', fontsize=9)
    ax.set_title('g-factor (TruncNormal)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    
    # 2. a_scale: HalfNormal(σ=2.0) + clip[0.1, 10]
    ax = axes[1]
    a_range = np.linspace(0.0, 12, 500)
    a_prior_raw = halfnorm.pdf(a_range, scale=2.0)
    # clip効果を近似的に表示
    a_prior = np.where((a_range >= 0.1) & (a_range <= 10.0), a_prior_raw, 0.0)
    ax.plot(a_range, a_prior, 'b-', lw=2, label='Prior (HalfNormal)')
    ax.axvline(v6_params['a'], color='r', linestyle='--', lw=1.5, label=f'v6: {v6_params["a"]:.2f}')
    ax.axvspan(0, 0.1, alpha=0.2, color='gray', label='Clipped')
    ax.axvspan(10, 12, alpha=0.2, color='gray')
    ax.set_xlabel('a', fontsize=9)
    ax.set_ylabel('Prob. Density', fontsize=9)
    ax.set_title('a (HalfNormal σ=2)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 12])
    
    # 3. B4: LogNormal(μ=log(2mK), σ=1.2) + clip[0.01mK, 50mK]
    ax = axes[2]
    B4_range = np.linspace(0.001, 60, 500)  # mK単位
    B4_log_mu = np.log(2.0)  # 2mK
    B4_log_sigma = 1.2
    B4_prior_raw = lognorm.pdf(B4_range, s=B4_log_sigma, scale=np.exp(B4_log_mu))
    B4_prior = np.where((B4_range >= 0.01) & (B4_range <= 50.0), B4_prior_raw, 0.0)
    ax.plot(B4_range, B4_prior, 'b-', lw=2, label='Prior (LogNormal)')
    ax.axvline(v6_params['B4'] * 1000, color='r', linestyle='--', lw=1.5, 
               label=f'v6: {v6_params["B4"]*1000:.1f}mK')
    ax.axvspan(50, 60, alpha=0.2, color='gray', label='Clipped')
    ax.set_xlabel('B₄ (mK)', fontsize=9)
    ax.set_ylabel('Prob. Density', fontsize=9)
    ax.set_title('B₄ (LogNormal μ=2mK)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 60])
    
    # 4. B6: Normal(μ=0, σ=0.5mK) + clip[-2mK, 2mK]
    ax = axes[3]
    B6_range = np.linspace(-2.5, 2.5, 500)  # mK単位
    B6_prior_raw = norm.pdf(B6_range, loc=0, scale=0.5)
    B6_prior = np.where((B6_range >= -2.0) & (B6_range <= 2.0), B6_prior_raw, 0.0)
    ax.plot(B6_range, B6_prior, 'b-', lw=2, label='Prior (Normal)')
    ax.axvline(v6_params['B6'] * 1000, color='r', linestyle='--', lw=1.5, 
               label=f'v6: {v6_params["B6"]*1000:.2f}mK')
    ax.axvspan(-2.5, -2.0, alpha=0.2, color='gray', label='Clipped')
    ax.axvspan(2.0, 2.5, alpha=0.2, color='gray')
    ax.set_xlabel('B₆ (mK)', fontsize=9)
    ax.set_ylabel('Prob. Density', fontsize=9)
    ax.set_title('B₆ (Normal σ=0.5mK)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    
    # 5. eps_bg: TruncatedNormal(μ=v6平均, σ=0.3, [13, 16])
    ax = axes[4]
    eps_range = np.linspace(12, 17, 500)
    eps_mu = v6_params['eps']  # v6平均値を中心
    eps_sigma = 0.3  # v7で0.5→0.3に変更
    a_trunc_eps = (13.0 - eps_mu) / eps_sigma
    b_trunc_eps = (16.0 - eps_mu) / eps_sigma
    eps_prior = truncnorm.pdf(eps_range, a_trunc_eps, b_trunc_eps, loc=eps_mu, scale=eps_sigma)
    ax.plot(eps_range, eps_prior, 'b-', lw=2, label='Prior')
    ax.axvline(eps_mu, color='r', linestyle='--', lw=1.5, label=f'v6: {eps_mu:.1f}')
    ax.set_xlabel('ε_bg', fontsize=9)
    ax.set_ylabel('Prob. Density', fontsize=9)
    ax.set_title('ε_bg (TruncNormal σ=0.3)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    
    # 6. γ_mean (階層パラメータ): TruncatedNormal(μ=0.074, σ=0.16, [0.005, 0.3])
    ax = axes[5]
    gamma_mean_range = np.linspace(0, 0.35, 500)
    gamma_mean_mu = GAMMA_HYPERPRIOR_MU  # 0.074
    gamma_mean_sigma = GAMMA_HYPERPRIOR_SIGMA  # 0.16
    a_trunc_gm = (0.005 - gamma_mean_mu) / gamma_mean_sigma
    b_trunc_gm = (0.3 - gamma_mean_mu) / gamma_mean_sigma
    gamma_mean_prior = truncnorm.pdf(gamma_mean_range, a_trunc_gm, b_trunc_gm, 
                                      loc=gamma_mean_mu, scale=gamma_mean_sigma)
    ax.plot(gamma_mean_range, gamma_mean_prior, 'b-', lw=2, label='Prior')
    # v6の非張り付きγ平均
    v6_gamma_nonbound = [g for g in v6_params['gamma'] if g > 0.015]
    if v6_gamma_nonbound:
        v6_gamma_mean = np.mean(v6_gamma_nonbound)
        ax.axvline(v6_gamma_mean, color='r', linestyle='--', lw=1.5, 
                   label=f'v6 mean: {v6_gamma_mean*1000:.0f}GHz')
    ax.set_xlabel('γ_mean (THz)', fontsize=9)
    ax.set_ylabel('Prob. Density', fontsize=9)
    ax.set_title('γ_mean (Hierarchical)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    
    # 7. γ_std (階層パラメータ): HalfNormal(σ=0.092)
    ax = axes[6]
    gamma_std_range = np.linspace(0, 0.15, 500)
    gamma_std_prior = halfnorm.pdf(gamma_std_range, scale=GAMMA_STD_PRIOR)  # 0.092
    ax.plot(gamma_std_range, gamma_std_prior, 'b-', lw=2, label='Prior (HalfNormal)')
    # v6のγ標準偏差
    if v6_gamma_nonbound:
        v6_gamma_std = np.std(v6_gamma_nonbound)
        ax.axvline(v6_gamma_std, color='r', linestyle='--', lw=1.5, 
                   label=f'v6 std: {v6_gamma_std*1000:.0f}GHz')
    ax.set_xlabel('γ_std (THz)', fontsize=9)
    ax.set_ylabel('Prob. Density', fontsize=9)
    ax.set_title('γ_std (Hierarchical)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    
    # 8-14. gamma_1 ~ gamma_7: 階層モデルの個別γ
    # 階層モデルではγ_meanとγ_stdに従うため、条件付き分布を表示
    gamma_range = np.linspace(0, 0.6, 500)
    # γ_mean=0.07, γ_std=0.03の場合の典型的な分布
    a_trunc_gamma = (0.005 - gamma_mean_mu) / (GAMMA_STD_PRIOR + 1e-6)
    b_trunc_gamma = (0.5 - gamma_mean_mu) / (GAMMA_STD_PRIOR + 1e-6)
    gamma_prior = truncnorm.pdf(gamma_range, a_trunc_gamma, b_trunc_gamma, 
                                 loc=gamma_mean_mu, scale=GAMMA_STD_PRIOR)
    
    for i in range(7):
        ax = axes[7 + i]
        ax.plot(gamma_range, gamma_prior, 'b-', lw=2, label='Prior (Hierarchical)')
        if i < len(v6_params['gamma']):
            ax.axvline(v6_params['gamma'][i], color='r', linestyle='--', lw=1.5, 
                      label=f'v6: {v6_params["gamma"][i]*1000:.0f}GHz')
        ax.set_xlabel(f'γ_{i+1} (THz)', fontsize=9)
        ax.set_ylabel('Prob. Density', fontsize=9)
        ax.set_title(f'γ_{i+1} (Pooled)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
    
    # 未使用のaxesを非表示
    for idx in range(14, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_dir:
        save_path = save_dir / f'prior_distributions_{model_form}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ prior_distributions_{model_form}.png saved")
    
    plt.close()


def plot_posterior_distributions(trace, model_form='H', save_dir=None):
    """事後分布の可視化（ArviZを使用）"""
    print(f"\n{'='*80}")
    print(f"事後分布プロット作成 ({model_form}-form)")
    print(f"{'='*80}")
    
    # 物理値への変換
    posterior = trace.posterior
    
    var_names_scaled = ['g_factor_scaled', 'a_scale_scaled', 'B4_scaled', 'B6_scaled', 'eps_bg_scaled']
    var_names_scaled += [f'gamma_{i+1}_scaled' for i in range(7)]
    
    # 1. トレースプロット（収束診断）
    fig = az.plot_trace(trace, var_names=var_names_scaled[:5], compact=True, figsize=(15, 12))
    fig[0, 0].figure.suptitle(f'Trace Plot - Global Parameters ({model_form}-form)', fontsize=14, y=0.995)
    if save_dir:
        plt.savefig(save_dir / f'trace_global_{model_form}.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ trace_global_{model_form}.png saved")
    plt.close()
    
    fig = az.plot_trace(trace, var_names=var_names_scaled[5:], compact=True, figsize=(15, 18))
    fig[0, 0].figure.suptitle(f'Trace Plot - Gamma Parameters ({model_form}-form)', fontsize=14, y=0.995)
    if save_dir:
        plt.savefig(save_dir / f'trace_gamma_{model_form}.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ trace_gamma_{model_form}.png saved")
    plt.close()
    
    # 2. 事後分布（物理値空間でプロット）
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Posterior Distributions ({model_form}-form)', fontsize=16, y=0.995)
    axes = axes.flatten()
    
    # 物理値への変換とプロット
    param_info = [
        ('g_factor_scaled', 'g', SCALING_FACTORS['g'], 'g-factor', ''),
        ('a_scale_scaled', 'a', SCALING_FACTORS['a'], 'a (coupling)', ''),
        ('B4_scaled', 'B4', SCALING_FACTORS['B4'], 'B₄', 'mK'),
        ('B6_scaled', 'B6', SCALING_FACTORS['B6'], 'B₆', 'mK'),
        ('eps_bg_scaled', 'eps', SCALING_FACTORS['eps'], 'ε_bg', ''),
    ]
    
    for i, (var_scaled, var_phys, scale_factor, label, unit) in enumerate(param_info):
        ax = axes[i]
        samples_scaled = posterior[var_scaled].values.flatten()
        samples_phys = samples_scaled / scale_factor
        
        # mK単位に変換
        if unit == 'mK':
            samples_phys = samples_phys * 1000
            xlabel = f'{label} ({unit})'
        else:
            xlabel = label
        
        ax.hist(samples_phys, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        mean_val = np.mean(samples_phys)
        median_val = np.median(samples_phys)
        hdi = az.hdi(samples_phys, hdi_prob=0.94)
        
        ax.axvline(mean_val, color='red', linestyle='--', lw=1.5, label=f'Mean: {mean_val:.3g}')
        ax.axvline(median_val, color='orange', linestyle='-.', lw=1.5, label=f'Med: {median_val:.3g}')
        ax.axvspan(hdi[0], hdi[1], alpha=0.2, color='green', label=f'94% HDI')
        
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'{label}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.8)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
    
    # gamma parameters
    for i in range(7):
        ax = axes[5 + i]
        var_name = f'gamma_{i+1}_scaled'
        samples_scaled = posterior[var_name].values.flatten()
        samples_phys = samples_scaled / SCALING_FACTORS['gamma']
        
        ax.hist(samples_phys, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        mean_val = np.mean(samples_phys)
        median_val = np.median(samples_phys)
        hdi = az.hdi(samples_phys, hdi_prob=0.94)
        
        ax.axvline(mean_val, color='red', linestyle='--', lw=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='-.', lw=1.5, label=f'Med: {median_val:.2f}')
        ax.axvspan(hdi[0], hdi[1], alpha=0.2, color='green', label=f'94% HDI')
        
        ax.set_xlabel(f'γ_{i+1} (THz)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'γ_{i+1}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.8)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / f'posterior_distributions_{model_form}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ posterior_distributions_{model_form}.png saved")
    
    plt.close()
    
    # 3. ペアプロット（パラメータ相関）
    print("  パラメータ相関プロット作成中...")
    pair_vars = ['g_factor_scaled', 'a_scale_scaled', 'B4_scaled', 'B6_scaled', 'eps_bg_scaled']
    axes_pair = az.plot_pair(trace, var_names=pair_vars, kind='kde', 
                             marginals=True, figsize=(14, 14))
    # az.plot_pairはaxesの配列を返すので、figureを取得
    if hasattr(axes_pair, 'flatten'):
        fig_pair = axes_pair.flatten()[0].figure
    else:
        fig_pair = axes_pair[0, 0].figure
    fig_pair.suptitle(f'Parameter Correlations ({model_form}-form)', fontsize=14, y=0.995)
    if save_dir:
        plt.savefig(save_dir / f'pair_plot_{model_form}.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ pair_plot_{model_form}.png saved")
    plt.close()


def plot_prior_posterior_comparison(trace, v6_params, model_form='H', save_dir=None):
    """
    事前分布と事後分布の比較プロット - v7階層モデル対応
    
    新しい事前分布を反映:
    - a: HalfNormal(σ=2)
    - B₄: LogNormal(μ=log(2mK), σ=1.2)
    - B₆: Normal(μ=0, σ=0.5mK)
    - γ: 階層モデル（γ_mean, γ_stdから生成）
    """
    from scipy.stats import halfnorm, lognorm, norm
    
    print(f"\n{'='*80}")
    print(f"事前分布 vs 事後分布 比較プロット作成 ({model_form}-form) [v7]")
    print(f"{'='*80}")
    
    posterior = trace.posterior
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 13))
    fig.suptitle(f'Prior vs Posterior Comparison ({model_form}-form) - v7 Hierarchical', fontsize=16, y=0.98)
    axes = axes.flatten()
    
    # 1. g_factor: TruncatedNormal(μ=2.0, σ=0.05)
    ax = axes[0]
    g_range = np.linspace(1.5, 2.8, 500)
    a_trunc, b_trunc = (1.5 - 2.0) / 0.05, (2.8 - 2.0) / 0.05
    g_prior = truncnorm.pdf(g_range, a_trunc, b_trunc, loc=2.0, scale=0.05)
    
    samples_g = posterior['g_factor_scaled'].values.flatten() / SCALING_FACTORS['g']
    ax.hist(samples_g, bins=50, density=True, alpha=0.6, color='steelblue', 
            edgecolor='black', linewidth=0.5, label='Posterior')
    ax.plot(g_range, g_prior, 'r-', lw=2, label='Prior')
    ax.axvline(v6_params['g'], color='orange', linestyle='--', lw=1.5, label=f'v6: {v6_params["g"]:.2f}')
    ax.set_xlabel('g-factor', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('g-factor (TruncNormal)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)
    
    # 2. a_scale: HalfNormal(σ=2.0) + clip[0.1, 10]
    ax = axes[1]
    a_range = np.linspace(0.0, 12, 500)
    a_prior_raw = halfnorm.pdf(a_range, scale=2.0)
    a_prior = np.where((a_range >= 0.1) & (a_range <= 10.0), a_prior_raw, 0.0)
    
    samples_a = posterior['a_scale_scaled'].values.flatten() / SCALING_FACTORS['a']
    ax.hist(samples_a, bins=50, density=True, alpha=0.6, color='steelblue', 
            edgecolor='black', linewidth=0.5, label='Posterior')
    ax.plot(a_range, a_prior, 'r-', lw=2, label='Prior')
    ax.axvline(v6_params['a'], color='orange', linestyle='--', lw=1.5, label=f'v6: {v6_params["a"]:.2f}')
    ax.set_xlabel('a', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('a (HalfNormal)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)
    
    # 3. B4: LogNormal(μ=log(2mK), σ=1.2) + clip[0.01mK, 50mK]
    ax = axes[2]
    B4_range = np.linspace(0.001, 60, 500)  # mK
    B4_log_mu = np.log(2.0)
    B4_log_sigma = 1.2
    B4_prior_raw = lognorm.pdf(B4_range, s=B4_log_sigma, scale=np.exp(B4_log_mu))
    B4_prior = np.where((B4_range >= 0.01) & (B4_range <= 50.0), B4_prior_raw, 0.0)
    
    samples_B4 = posterior['B4_scaled'].values.flatten() / SCALING_FACTORS['B4'] * 1000  # mK
    ax.hist(samples_B4, bins=50, density=True, alpha=0.6, color='steelblue', 
            edgecolor='black', linewidth=0.5, label='Posterior')
    ax.plot(B4_range, B4_prior, 'r-', lw=2, label='Prior')
    ax.axvline(v6_params['B4'] * 1000, color='orange', linestyle='--', lw=1.5, 
               label=f'v6: {v6_params["B4"]*1000:.1f}mK')
    ax.set_xlabel('B₄ (mK)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('B₄ (LogNormal)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)
    
    # 4. B6: Normal(μ=0, σ=0.5mK) + clip[-2mK, 2mK]
    ax = axes[3]
    B6_range = np.linspace(-2.5, 2.5, 500)  # mK
    B6_prior_raw = norm.pdf(B6_range, loc=0, scale=0.5)
    B6_prior = np.where((B6_range >= -2.0) & (B6_range <= 2.0), B6_prior_raw, 0.0)
    
    samples_B6 = posterior['B6_scaled'].values.flatten() / SCALING_FACTORS['B6'] * 1000  # mK
    ax.hist(samples_B6, bins=50, density=True, alpha=0.6, color='steelblue', 
            edgecolor='black', linewidth=0.5, label='Posterior')
    ax.plot(B6_range, B6_prior, 'r-', lw=2, label='Prior')
    ax.axvline(v6_params['B6'] * 1000, color='orange', linestyle='--', lw=1.5, 
               label=f'v6: {v6_params["B6"]*1000:.2f}mK')
    ax.set_xlabel('B₆ (mK)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('B₆ (Normal)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)
    
    # 5. eps_bg: TruncatedNormal(μ=v6平均, σ=0.3)
    ax = axes[4]
    eps_range = np.linspace(12, 17, 500)
    eps_mu = v6_params['eps']
    eps_sigma = 0.3  # v7: 0.5→0.3
    a_trunc_eps = (13.0 - eps_mu) / eps_sigma
    b_trunc_eps = (16.0 - eps_mu) / eps_sigma
    eps_prior = truncnorm.pdf(eps_range, a_trunc_eps, b_trunc_eps, loc=eps_mu, scale=eps_sigma)
    
    samples_eps = posterior['eps_bg_scaled'].values.flatten() / SCALING_FACTORS['eps']
    ax.hist(samples_eps, bins=50, density=True, alpha=0.6, color='steelblue', 
            edgecolor='black', linewidth=0.5, label='Posterior')
    ax.plot(eps_range, eps_prior, 'r-', lw=2, label='Prior')
    ax.axvline(eps_mu, color='orange', linestyle='--', lw=1.5, label=f'v6: {eps_mu:.1f}')
    ax.set_xlabel('ε_bg', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('ε_bg (TruncNormal σ=0.3)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)
    
    # 6. γ_mean (階層パラメータ)
    ax = axes[5]
    gamma_mean_range = np.linspace(0, 0.35, 500)
    gamma_mean_mu = GAMMA_HYPERPRIOR_MU
    gamma_mean_sigma = GAMMA_HYPERPRIOR_SIGMA
    a_trunc_gm = (0.005 - gamma_mean_mu) / gamma_mean_sigma
    b_trunc_gm = (0.3 - gamma_mean_mu) / gamma_mean_sigma
    gamma_mean_prior = truncnorm.pdf(gamma_mean_range, a_trunc_gm, b_trunc_gm,
                                      loc=gamma_mean_mu, scale=gamma_mean_sigma)
    
    samples_gamma_mean = posterior['gamma_mean_scaled'].values.flatten() / SCALING_FACTORS['gamma']
    ax.hist(samples_gamma_mean, bins=50, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5, label='Posterior')
    ax.plot(gamma_mean_range, gamma_mean_prior, 'r-', lw=2, label='Prior')
    ax.set_xlabel('γ_mean (THz)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('γ_mean (Hierarchical)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)
    
    # 7. γ_std (階層パラメータ)
    ax = axes[6]
    gamma_std_range = np.linspace(0, 0.3, 500)
    gamma_std_prior = halfnorm.pdf(gamma_std_range, scale=GAMMA_STD_PRIOR)
    
    samples_gamma_std = posterior['gamma_std_scaled'].values.flatten() / SCALING_FACTORS['gamma']
    ax.hist(samples_gamma_std, bins=50, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5, label='Posterior')
    ax.plot(gamma_std_range, gamma_std_prior, 'r-', lw=2, label='Prior')
    ax.set_xlabel('γ_std (THz)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('γ_std (Hierarchical)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)
    
    # 8-14. gamma_1 ~ gamma_7: 階層モデルの個別γ
    gamma_range = np.linspace(0, 0.6, 500)
    # 階層モデルの事前分布（WNLSベース: γ_mean=0.074, γ_std=0.092）
    a_trunc_gamma = (0.005 - GAMMA_HYPERPRIOR_MU) / (GAMMA_STD_PRIOR + 1e-6)
    b_trunc_gamma = (0.5 - GAMMA_HYPERPRIOR_MU) / (GAMMA_STD_PRIOR + 1e-6)
    gamma_prior = truncnorm.pdf(gamma_range, a_trunc_gamma, b_trunc_gamma,
                                 loc=GAMMA_HYPERPRIOR_MU, scale=GAMMA_STD_PRIOR)
    
    for i in range(7):
        ax = axes[7 + i]
        var_name = f'gamma_{i+1}_scaled'
        samples_gamma = posterior[var_name].values.flatten() / SCALING_FACTORS['gamma']
        
        ax.hist(samples_gamma, bins=50, density=True, alpha=0.6, color='steelblue', 
                edgecolor='black', linewidth=0.5, label='Posterior')
        ax.plot(gamma_range, gamma_prior, 'r-', lw=2, label='Prior')
        if i < len(v6_params['gamma']):
            ax.axvline(v6_params['gamma'][i], color='orange', linestyle='--', lw=1.5, 
                      label=f'v6: {v6_params["gamma"][i]*1000:.0f}GHz')
        ax.set_xlabel(f'γ_{i+1} (THz)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'γ_{i+1} (Pooled)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.8)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
    
    # 未使用のaxesを非表示
    for idx in range(14, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_dir:
        save_path = save_dir / f'prior_posterior_comparison_{model_form}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ prior_posterior_comparison_{model_form}.png saved")
    
    plt.close()


def plot_posterior_predictive_spectra(trace, datasets, v6_params, model_form='H', save_dir=None, n_samples=500):
    """事後予測透過スペクトルのプロット（95% HDI区間 + 中央値）"""
    print(f"\n{'='*80}")
    print(f"事後予測透過スペクトルプロット作成 ({model_form}-form)")
    print(f"{'='*80}")
    
    posterior = trace.posterior
    
    # 事後分布からランダムサンプリング
    n_chains = posterior.dims['chain']
    n_draws = posterior.dims['draw']
    total_samples = n_chains * n_draws
    
    # サンプル数を制限（計算時間短縮）
    if total_samples > n_samples:
        sample_indices = np.random.choice(total_samples, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
        n_samples = total_samples
    
    print(f"  事後分布から {n_samples} サンプルを使用")
    
    # プロット準備
    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig.suptitle(f'Posterior Predictive Transmission Spectra ({model_form}-form)', 
                 fontsize=14, y=0.995)
    axes = axes.flatten()
    
    for idx, data in enumerate(datasets):
        ax = axes[idx]
        freq = data['freq']
        trans_obs = data['trans']
        B = data['B']
        T = data['T']
        label = data['label']
        
        print(f"  計算中: {label} (B={B}T, T={T}K)")
        
        # 各サンプルで透過スペクトルを計算
        trans_samples = np.zeros((n_samples, len(freq)))
        
        for i, sample_idx in enumerate(sample_indices):
            chain_idx = sample_idx // n_draws
            draw_idx = sample_idx % n_draws
            
            # パラメータ取得（スケーリング解除）
            g = float(posterior['g_factor_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['g']
            a = float(posterior['a_scale_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['a']
            B4 = float(posterior['B4_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B4']
            B6 = float(posterior['B6_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B6']
            eps = float(posterior['eps_bg_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['eps']
            
            gamma_array = np.array([
                float(posterior[f'gamma_{j+1}_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            
            # 透過スペクトル計算
            trans_samples[i, :] = calculate_transmission_for_params(
                freq, B, T, g, a, B4, B6, eps, gamma_array, model_form
            )
        
        # 統計量計算
        trans_median = np.median(trans_samples, axis=0)
        trans_hdi = az.hdi(trans_samples, hdi_prob=0.95)
        
        # プロット
        ax.plot(freq, trans_obs, 'ko', markersize=3, alpha=0.6, label='Observed')
        ax.plot(freq, trans_median, 'r-', lw=2, label='Median')
        ax.fill_between(freq, trans_hdi[:, 0], trans_hdi[:, 1], 
                        color='red', alpha=0.2, label='95% HDI')
        
        ax.set_xlabel('Frequency (THz)', fontsize=10)
        ax.set_ylabel('Transmission', fontsize=10)
        ax.set_title(f'{label} (B={B}T, T={T}K)', fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([freq.min(), freq.max()])
        ax.set_ylim([0, 1])
    
    # 未使用のaxesを非表示
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / f'posterior_predictive_spectra_{model_form}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ posterior_predictive_spectra_{model_form}.png saved")
    
    plt.close()
    
    # 残差プロット（観測値 - 中央値）
    print(f"\n  残差プロット作成中...")
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig.suptitle(f'Posterior Predictive Residuals ({model_form}-form)', 
                 fontsize=14, y=0.995)
    axes = axes.flatten()
    
    for idx, data in enumerate(datasets):
        ax = axes[idx]
        freq = data['freq']
        trans_obs = data['trans']
        B = data['B']
        T = data['T']
        label = data['label']
        
        # 各サンプルで透過スペクトルを計算（再計算）
        trans_samples = np.zeros((n_samples, len(freq)))
        
        for i, sample_idx in enumerate(sample_indices):
            chain_idx = sample_idx // n_draws
            draw_idx = sample_idx % n_draws
            
            g = float(posterior['g_factor_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['g']
            a = float(posterior['a_scale_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['a']
            B4 = float(posterior['B4_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B4']
            B6 = float(posterior['B6_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B6']
            eps = float(posterior['eps_bg_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['eps']
            
            gamma_array = np.array([
                float(posterior[f'gamma_{j+1}_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            
            trans_samples[i, :] = calculate_transmission_for_params(
                freq, B, T, g, a, B4, B6, eps, gamma_array, model_form
            )
        
        trans_median = np.median(trans_samples, axis=0)
        residual = trans_obs - trans_median
        
        ax.plot(freq, residual, 'ko-', markersize=3, lw=1, alpha=0.6)
        ax.axhline(0, color='red', linestyle='--', lw=1.5)
        ax.set_xlabel('Frequency (THz)', fontsize=10)
        ax.set_ylabel('Residual (Obs - Pred)', fontsize=10)
        ax.set_title(f'{label} (RMSE={np.sqrt(np.mean(residual**2)):.4f})', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim([freq.min(), freq.max()])
    
    # 未使用のaxesを非表示
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / f'posterior_predictive_residuals_{model_form}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ posterior_predictive_residuals_{model_form}.png saved")
    
    plt.close()


def plot_posterior_predictive_spectra_comparison(trace_H, trace_B, datasets, save_dir=None, n_samples=500):
    """
    H形式とB形式の事後予測透過スペクトルを1枚のグラフに重ねてプロット
    
    - H形式: 赤色
    - B形式: 青色  
    - ポラリトン領域: オレンジ色で塗りつぶし（重み2.0×）
    - 共振器領域: 緑色で塗りつぶし（重み1.0×）
    - それ以外: 塗りつぶしなし（重み0.01）
    """
    print(f"\n{'='*80}")
    print(f"事後予測透過スペクトル比較プロット作成 (H vs B)")
    print(f"{'='*80}")
    
    posterior_H = trace_H.posterior
    posterior_B = trace_B.posterior
    
    # 事後分布からランダムサンプリング
    n_chains_H = posterior_H.dims['chain']
    n_draws_H = posterior_H.dims['draw']
    total_samples_H = n_chains_H * n_draws_H
    
    n_chains_B = posterior_B.dims['chain']
    n_draws_B = posterior_B.dims['draw']
    total_samples_B = n_chains_B * n_draws_B
    
    # サンプル数を制限
    if total_samples_H > n_samples:
        sample_indices_H = np.random.choice(total_samples_H, size=n_samples, replace=False)
    else:
        sample_indices_H = np.arange(total_samples_H)
        n_samples = total_samples_H
        
    if total_samples_B > n_samples:
        sample_indices_B = np.random.choice(total_samples_B, size=n_samples, replace=False)
    else:
        sample_indices_B = np.arange(total_samples_B)
    
    print(f"  事後分布から {n_samples} サンプルを使用")
    
    # プロット準備
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle('Posterior Predictive Transmission Spectra: H-form (red) vs B-form (blue)', 
                 fontsize=14, fontweight='bold', y=0.995)
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    for idx, data in enumerate(datasets):
        ax = axes[idx]
        freq = data['freq']
        trans_obs = data['trans']
        B = data['B']
        T = data['T']
        label = data['label']
        
        print(f"  計算中: {label} (B={B}T, T={T}K)")
        
        # H形式: 各サンプルで透過スペクトルを計算
        trans_samples_H = np.zeros((n_samples, len(freq)))
        
        for i, sample_idx in enumerate(sample_indices_H):
            chain_idx = sample_idx // n_draws_H
            draw_idx = sample_idx % n_draws_H
            
            g = float(posterior_H['g_factor_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['g']
            a = float(posterior_H['a_scale_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['a']
            B4 = float(posterior_H['B4_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B4']
            B6 = float(posterior_H['B6_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B6']
            eps = float(posterior_H['eps_bg_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['eps']
            
            gamma_array = np.array([
                float(posterior_H[f'gamma_{j+1}_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            
            trans_samples_H[i, :] = calculate_transmission_for_params(
                freq, B, T, g, a, B4, B6, eps, gamma_array, model_form='H'
            )
        
        # B形式: 各サンプルで透過スペクトルを計算
        trans_samples_B = np.zeros((n_samples, len(freq)))
        
        for i, sample_idx in enumerate(sample_indices_B):
            chain_idx = sample_idx // n_draws_B
            draw_idx = sample_idx % n_draws_B
            
            g = float(posterior_B['g_factor_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['g']
            a = float(posterior_B['a_scale_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['a']
            B4 = float(posterior_B['B4_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B4']
            B6 = float(posterior_B['B6_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B6']
            eps = float(posterior_B['eps_bg_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['eps']
            
            gamma_array = np.array([
                float(posterior_B[f'gamma_{j+1}_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            
            trans_samples_B[i, :] = calculate_transmission_for_params(
                freq, B, T, g, a, B4, B6, eps, gamma_array, model_form='B'
            )
        
        # 統計量計算
        trans_median_H = np.median(trans_samples_H, axis=0)
        trans_hdi_H = az.hdi(trans_samples_H, hdi_prob=0.95)
        trans_median_B = np.median(trans_samples_B, axis=0)
        trans_hdi_B = az.hdi(trans_samples_B, hdi_prob=0.95)
        
        # ポラリトン/共振器領域の検出（H形式基準）
        polariton_regions, cavity_regions = detect_peaks_and_classify(freq, trans_median_H)
        
        # 領域の塗りつぶし
        polariton_legend_added = False
        for freq_start, freq_end in polariton_regions:
            label_region = 'Polariton (2.0×)' if not polariton_legend_added else None
            ax.axvspan(freq_start, freq_end, alpha=0.12, color='orange', label=label_region, zorder=1)
            polariton_legend_added = True
        
        cavity_legend_added = False
        for freq_start, freq_end in cavity_regions:
            label_region = 'Cavity (1.0×)' if not cavity_legend_added else None
            ax.axvspan(freq_start, freq_end, alpha=0.12, color='green', label=label_region, zorder=1)
            cavity_legend_added = True
        
        # データプロット
        ax.plot(freq, trans_obs, 'o', color='gray', markersize=2.5, alpha=0.6, 
                label='Data', zorder=2)
        
        # H形式（赤）
        ax.plot(freq, trans_median_H, '-', color='red', linewidth=2.0, 
                label='H-form Median', zorder=4)
        ax.fill_between(freq, trans_hdi_H[:, 0], trans_hdi_H[:, 1], 
                        color='red', alpha=0.15, label='H-form 95% HDI', zorder=3)
        
        # B形式（青）
        ax.plot(freq, trans_median_B, '-', color='blue', linewidth=2.0, 
                label='B-form Median', zorder=4)
        ax.fill_between(freq, trans_hdi_B[:, 0], trans_hdi_B[:, 1], 
                        color='blue', alpha=0.15, label='B-form 95% HDI', zorder=3)
        
        # RMSE計算
        rmse_H = np.sqrt(np.mean((trans_obs - trans_median_H)**2))
        rmse_B = np.sqrt(np.mean((trans_obs - trans_median_B)**2))
        
        ax.set_title(f"{label}\nH-RMSE: {rmse_H:.4f}, B-RMSE: {rmse_B:.4f}", 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (THz)', fontsize=10)
        ax.set_ylabel('Transmittance', fontsize=10)
        ax.legend(fontsize=6, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
        
        # y軸範囲の自動調整
        y_margin = 0.05
        y_min = min(np.min(trans_obs), np.min(trans_median_H), np.min(trans_median_B)) - y_margin
        y_max = max(np.max(trans_obs), np.max(trans_median_H), np.max(trans_median_B)) + y_margin
        ax.set_ylim(max(0, y_min), min(1.1, y_max))
        ax.set_xlim([freq.min(), freq.max()])
    
    # 未使用のaxesを非表示
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'posterior_predictive_spectra_HB_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ posterior_predictive_spectra_HB_comparison.png saved")
    
    plt.close()
    
    # 残差比較プロット
    print(f"\n  残差比較プロット作成中...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle('Posterior Predictive Residuals: H-form (red) vs B-form (blue)', 
                 fontsize=14, fontweight='bold', y=0.995)
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    for idx, data in enumerate(datasets):
        ax = axes[idx]
        freq = data['freq']
        trans_obs = data['trans']
        B = data['B']
        T = data['T']
        label = data['label']
        
        # 中央値を再計算（前のループで計算済みの場合は保存しておくべきだが、簡略化のため再計算）
        trans_samples_H = np.zeros((n_samples, len(freq)))
        trans_samples_B = np.zeros((n_samples, len(freq)))
        
        for i, sample_idx in enumerate(sample_indices_H):
            chain_idx = sample_idx // n_draws_H
            draw_idx = sample_idx % n_draws_H
            g = float(posterior_H['g_factor_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['g']
            a = float(posterior_H['a_scale_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['a']
            B4 = float(posterior_H['B4_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B4']
            B6 = float(posterior_H['B6_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B6']
            eps = float(posterior_H['eps_bg_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['eps']
            gamma_array = np.array([
                float(posterior_H[f'gamma_{j+1}_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            trans_samples_H[i, :] = calculate_transmission_for_params(
                freq, B, T, g, a, B4, B6, eps, gamma_array, model_form='H'
            )
        
        for i, sample_idx in enumerate(sample_indices_B):
            chain_idx = sample_idx // n_draws_B
            draw_idx = sample_idx % n_draws_B
            g = float(posterior_B['g_factor_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['g']
            a = float(posterior_B['a_scale_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['a']
            B4 = float(posterior_B['B4_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B4']
            B6 = float(posterior_B['B6_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['B6']
            eps = float(posterior_B['eps_bg_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['eps']
            gamma_array = np.array([
                float(posterior_B[f'gamma_{j+1}_scaled'].values[chain_idx, draw_idx]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            trans_samples_B[i, :] = calculate_transmission_for_params(
                freq, B, T, g, a, B4, B6, eps, gamma_array, model_form='B'
            )
        
        trans_median_H = np.median(trans_samples_H, axis=0)
        trans_median_B = np.median(trans_samples_B, axis=0)
        
        residual_H = trans_obs - trans_median_H
        residual_B = trans_obs - trans_median_B
        
        ax.plot(freq, residual_H, 'o-', color='red', markersize=2, lw=1, alpha=0.7, label='H-form')
        ax.plot(freq, residual_B, 'o-', color='blue', markersize=2, lw=1, alpha=0.7, label='B-form')
        ax.axhline(0, color='gray', linestyle='--', lw=1.5)
        
        rmse_H = np.sqrt(np.mean(residual_H**2))
        rmse_B = np.sqrt(np.mean(residual_B**2))
        
        ax.set_xlabel('Frequency (THz)', fontsize=10)
        ax.set_ylabel('Residual (Obs - Pred)', fontsize=10)
        ax.set_title(f'{label}\nH-RMSE: {rmse_H:.4f}, B-RMSE: {rmse_B:.4f}', fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([freq.min(), freq.max()])
    
    # 未使用のaxesを非表示
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'posterior_predictive_residuals_HB_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ posterior_predictive_residuals_HB_comparison.png saved")
    
    plt.close()


# ============================================================================
# デバッグモード設定（v7.1追加）
# ============================================================================
DEBUG_MODE = False  # True: クイックテスト（2データセット、500サンプル）

# ============================================================================
# メイン処理
# ============================================================================
def main():
    global TARGET_DATA, SMC_DRAWS, SMC_CHAINS
    
    start_time = time.time()
    
    # デバッグモード設定
    if DEBUG_MODE:
        print("\n" + "🔧"*40)
        print("デバッグモード: ON")
        print("  - データセット: 最初の2個のみ")
        print("  - SMC Draws (H): 500")
        print("  - SMC Draws (B): 1000")
        print("  - SMC Chains: 2")
        print("🔧"*40 + "\n")
        TARGET_DATA = TARGET_DATA[:2]
        SMC_DRAWS = 500
        SMC_CHAINS = 2
    
    print(f"\n{'='*80}")
    print(f"Bayesian Analysis with Parameter Scaling and LOO-CV (v7.1: Non-centered)")
    print(f"{'='*80}")
    print(f"【新機能】")
    print(f"  1. パラメータスケーリング（pre_test_v6準拠）")
    print(f"  2. LOO-CV (Leave-One-Out Cross-Validation) モデル評価")
    print(f"  3. 条件数改善による数値安定性向上")
    print(f"{'='*80}")
    
    # スケーリング係数の表示
    print(f"\n📊 パラメータスケーリング係数:")
    for key, value in SCALING_FACTORS.items():
        print(f"  {key}: {value}")
    
    # v6最適化結果の読み込み（H形式とB形式）
    print(f"\n{'='*80}")
    print("v6最適化結果の読み込み")
    print(f"{'='*80}")
    v6_params_H = load_v6_optimized_params('H')
    v6_params_B = load_v6_optimized_params('B')
    
    if v6_params_H is None or v6_params_B is None:
        print("❌ 最適化結果の読み込みに失敗しました")
        return
    
    # 結果ディレクトリ
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = pathlib.Path(__file__).parent / f"bayesian_results_scaled_loocv_{timestamp}"
    results_dir.mkdir(exist_ok=True)
    print(f"\n📁 結果保存: {results_dir}")
    
    # データロード
    datasets = load_all_datasets(TARGET_DATA)
    if not datasets:
        print("❌ データがありません")
        return
    
    # 観測データと重み
    trans_obs_concat = np.concatenate([d['trans'] for d in datasets])
    weight_concat = np.concatenate([d['weight'] for d in datasets])
    sigma_eff = 0.01 / np.sqrt(weight_concat)
    
    # ============================================================================
    # H形式でベイズモデル構築（スケーリング版）
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"ベイズモデル構築（H形式、スケーリング対応）")
    print(f"{'='*80}")
    
    # スケーリングされた事前分布パラメータ
    print(f"\n📊 H形式事前分布（スケーリング空間）:")
    g_scaled_mu = v6_params_H['g'] * SCALING_FACTORS['g']
    a_scaled_mu = v6_params_H['a'] * SCALING_FACTORS['a']
    B4_scaled_mu = v6_params_H['B4'] * SCALING_FACTORS['B4']
    B6_scaled_mu = v6_params_H['B6'] * SCALING_FACTORS['B6']
    eps_scaled_mu = v6_params_H['eps'] * SCALING_FACTORS['eps']
    gamma_scaled_mu = v6_params_H['gamma'] * SCALING_FACTORS['gamma']
    
    print(f"  g_scaled = {g_scaled_mu:.4f} (物理値: {v6_params_H['g']:.4f})")
    print(f"  a_scaled = {a_scaled_mu:.4f} (物理値: {v6_params_H['a']:.4f})")
    print(f"  B4_scaled = {B4_scaled_mu:.4f} (物理値: {v6_params_H['B4']:.6f} K)")
    print(f"  B6_scaled = {B6_scaled_mu:.4f} (物理値: {v6_params_H['B6']:.6f} K)")
    print(f"  eps_scaled = {eps_scaled_mu:.4f} (物理値: {v6_params_H['eps']:.4f})")
    
    with pm.Model() as model_H:
        # ============================================================
        # 事前分布設定（物理的制約ベース、v7階層モデル）
        # ============================================================
        
        # ------------------------------
        # 1. g因子: 強情報（理論値準拠）
        # ------------------------------
        g_factor_scaled_H = pm.TruncatedNormal('g_factor_scaled',
            mu=2.0 * SCALING_FACTORS['g'],      # 理論値
            sigma=0.05 * SCALING_FACTORS['g'],  # 強情報
            lower=1.5 * SCALING_FACTORS['g'],
            upper=2.8 * SCALING_FACTORS['g'])
        
        # ------------------------------
        # 2. a: HalfNormal（低値優先、上限拡張）
        # ------------------------------
        # v6で a=5.0 張り付き → 上限を10に拡張 + 低値優先分布
        a_raw_H = pm.HalfNormal('a_raw_H', sigma=2.0)  # σ=2で99%が0-6に収まる
        a_scale_scaled_H = pm.Deterministic('a_scale_scaled',
            pt.clip(a_raw_H, 0.1, 10.0) * SCALING_FACTORS['a'])
        
        # ------------------------------
        # 3. B₄: LogNormal（正値保証、低値優先）
        # ------------------------------
        # H形式: v6で30mK張り付き → 上限50mKに拡張
        # LogNormal(μ_log, σ_log)でμ=2mK, 95%区間≈[0.2, 20]mK
        B4_log_mu = np.log(0.002)  # 2mK（対数平均）
        B4_log_sigma = 1.2         # 対数標準偏差
        B4_raw_H = pm.LogNormal('B4_raw_H', mu=B4_log_mu, sigma=B4_log_sigma)
        B4_scaled_H = pm.Deterministic('B4_scaled',
            pt.clip(B4_raw_H, 0.00001, 0.05) * SCALING_FACTORS['B4'])  # [0.01mK, 50mK]
        
        # ------------------------------
        # 4. B₆: Normal（ゼロ中心、対称）
        # ------------------------------
        # v6結果: H=-1.0mK, B=-1.0mK → ほぼ下限
        # Normal(0, 0.5mK)で95%区間≈[-1mK, +1mK]、範囲拡張
        B6_raw_H = pm.Normal('B6_raw_H', mu=0, sigma=0.0005)
        B6_scaled_H = pm.Deterministic('B6_scaled',
            pt.clip(B6_raw_H, -0.002, 0.002) * SCALING_FACTORS['B6'])  # [-2mK, +2mK]
        
        # ------------------------------
        # 5. ε_bg: 中情報（文献範囲+v6参考）
        # ------------------------------
        # v6: H=14.0, B=14.1 → 両方とも妥当な範囲
        eps_v6_avg = (v6_params_H['eps'] + v6_params_B['eps']) / 2  # v6平均値を参考
        eps_bg_scaled_H = pm.TruncatedNormal('eps_bg_scaled',
            mu=eps_v6_avg * SCALING_FACTORS['eps'],
            sigma=0.3 * SCALING_FACTORS['eps'],  # 0.5 → 0.3（情報強化）
            lower=13.0 * SCALING_FACTORS['eps'],
            upper=16.0 * SCALING_FACTORS['eps'])
        
        # ------------------------------
        # 6. γ: Non-centered階層モデル（識別不能性解消 + 収束性改善）
        # ------------------------------
        # v7.1: Non-centered Parameterizationで「漏斗」問題を解消
        # Centered版ではγ_meanが変化すると全γ_iの条件付き分布が連動変化
        # Non-centered版ではz_iとγ_meanが独立で、HMC/SMCが効率的に探索可能
        
        # ハイパーパラメータ（log空間で定義 → 正値保証）
        log_gamma_mu_H = pm.Normal('log_gamma_mu',
            mu=np.log(GAMMA_HYPERPRIOR_MU),  # log(0.074) ≈ -2.6
            sigma=0.3)  # log空間で緩めの事前分布
        
        log_gamma_sd_H = pm.HalfNormal('log_gamma_sd', sigma=0.3)
        
        # ★★★ Non-centered変換の核心 ★★★
        # 標準正規分布からサンプリング（上位パラメータと独立）
        gamma_raw_H = pm.Normal('gamma_raw', mu=0, sigma=1, shape=7)
        
        # 決定論的変換で物理値に変換
        # log(γ_i) = log(γ_mu) + log(γ_sd) * z_i
        # γ_i = exp(log(γ_mu) + log(γ_sd) * z_i)
        gamma_vec_unscaled_H = pm.Deterministic('gamma_vec',
            pt.exp(log_gamma_mu_H + log_gamma_sd_H * gamma_raw_H))
        
        # 切り捨て処理（物理的に意味のある範囲）+ スケーリング
        gamma_vec_scaled_H = pm.Deterministic('gamma_vec_scaled',
            pt.clip(gamma_vec_unscaled_H, 0.005, 0.5) * SCALING_FACTORS['gamma'])
        
        # 後方互換性のため個別gamma_i_scaledも定義
        for i in range(7):
            pm.Deterministic(f'gamma_{i+1}_scaled', gamma_vec_scaled_H[i])
        
        # 階層パラメータを物理値空間で記録（診断用）
        gamma_mean_scaled_H = pm.Deterministic('gamma_mean_scaled',
            pt.exp(log_gamma_mu_H) * SCALING_FACTORS['gamma'])
        gamma_std_scaled_H = pm.Deterministic('gamma_std_scaled',
            log_gamma_sd_H * SCALING_FACTORS['gamma'])
        
        # ------------------------------
        # 7. 尤度: StudentT（外れ値頑健）
        # ------------------------------
        model_op_H = ScaledInformedPriorModelOp(datasets, 'H')
        trans_pred_H = model_op_H(a_scale_scaled_H, gamma_vec_scaled_H,
                                   g_factor_scaled_H, B4_scaled_H,
                                   B6_scaled_H, eps_bg_scaled_H)
        
        likelihood_H = pm.StudentT('likelihood',
            nu=NU_STUDENTT,
            mu=trans_pred_H,
            sigma=sigma_eff,
            observed=trans_obs_concat)
        
        # ------------------------------
        # 8. SMCサンプリング
        # ------------------------------
        print(f"\n🔬 SMCサンプリング開始（H形式）")
        print(f"   Sampler: SMC (Sequential Monte Carlo)")
        print(f"   Draws: {SMC_DRAWS}, Chains: {SMC_CHAINS}, Cores: {SMC_CHAINS if SMC_PARALLEL else 1}")
        print(f"   Hierarchical γ: ON, Likelihood: StudentT(ν={NU_STUDENTT})")
        
        trace_H = pm.sample_smc(
            draws=SMC_DRAWS,
            chains=SMC_CHAINS,
            cores=SMC_CHAINS if SMC_PARALLEL else 1,
            return_inferencedata=True,
            progressbar=True,
            random_seed=RANDOM_SEED,
        )
        
        # SMCは自動的にlog_likelihoodを保存しないため、明示的に計算
        print(f"\n📊 log_likelihood計算中（H形式）...")
        pm.compute_log_likelihood(trace_H, model=model_H)
    
    print(f"\n✅ H形式サンプリング完了（log_likelihood保存済み）")
    
    # ============================================================================
    # B形式でベイズモデル構築（スケーリング版）
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"ベイズモデル構築（B形式、スケーリング対応）")
    print(f"{'='*80}")
    
    # スケーリングされた事前分布パラメータ
    print(f"\n📊 B形式事前分布（スケーリング空間）:")
    g_scaled_mu_B = v6_params_B['g'] * SCALING_FACTORS['g']
    a_scaled_mu_B = v6_params_B['a'] * SCALING_FACTORS['a']
    B4_scaled_mu_B = v6_params_B['B4'] * SCALING_FACTORS['B4']
    B6_scaled_mu_B = v6_params_B['B6'] * SCALING_FACTORS['B6']
    eps_scaled_mu_B = v6_params_B['eps'] * SCALING_FACTORS['eps']
    gamma_scaled_mu_B = v6_params_B['gamma'] * SCALING_FACTORS['gamma']
    
    print(f"  g_scaled = {g_scaled_mu_B:.4f} (物理値: {v6_params_B['g']:.4f})")
    print(f"  a_scaled = {a_scaled_mu_B:.4f} (物理値: {v6_params_B['a']:.4f})")
    print(f"  B4_scaled = {B4_scaled_mu_B:.4f} (物理値: {v6_params_B['B4']:.6f} K)")
    print(f"  B6_scaled = {B6_scaled_mu_B:.4f} (物理値: {v6_params_B['B6']:.6f} K)")
    print(f"  eps_scaled = {eps_scaled_mu_B:.4f} (物理値: {v6_params_B['eps']:.4f})")
    
    with pm.Model() as model_B:
        # ============================================================
        # 事前分布設定（物理的制約ベース、v7階層モデル）- B形式
        # ============================================================
        
        # ------------------------------
        # 1. g因子: 強情報（理論値準拠）
        # ------------------------------
        g_factor_scaled_B = pm.TruncatedNormal('g_factor_scaled',
            mu=2.0 * SCALING_FACTORS['g'],      # 理論値
            sigma=0.05 * SCALING_FACTORS['g'],  # 強情報
            lower=1.5 * SCALING_FACTORS['g'],
            upper=2.8 * SCALING_FACTORS['g'])
        
        # ------------------------------
        # 2. a: HalfNormal（低値優先、上限拡張）
        # ------------------------------
        a_raw_B = pm.HalfNormal('a_raw_B', sigma=2.0)
        a_scale_scaled_B = pm.Deterministic('a_scale_scaled',
            pt.clip(a_raw_B, 0.1, 10.0) * SCALING_FACTORS['a'])
        
        # ------------------------------
        # 3. B₄: LogNormal（正値保証、低値優先）
        # ------------------------------
        B4_log_mu_B = np.log(0.002)
        B4_log_sigma_B = 1.2
        B4_raw_B = pm.LogNormal('B4_raw_B', mu=B4_log_mu_B, sigma=B4_log_sigma_B)
        B4_scaled_B = pm.Deterministic('B4_scaled',
            pt.clip(B4_raw_B, 0.00001, 0.05) * SCALING_FACTORS['B4'])
        
        # ------------------------------
        # 4. B₆: Normal（ゼロ中心、対称）
        # ------------------------------
        B6_raw_B = pm.Normal('B6_raw_B', mu=0, sigma=0.0005)
        B6_scaled_B = pm.Deterministic('B6_scaled',
            pt.clip(B6_raw_B, -0.002, 0.002) * SCALING_FACTORS['B6'])
        
        # ------------------------------
        # 5. ε_bg: 中情報（文献範囲+v6参考）
        # ------------------------------
        eps_v6_avg_B = (v6_params_H['eps'] + v6_params_B['eps']) / 2
        eps_bg_scaled_B = pm.TruncatedNormal('eps_bg_scaled',
            mu=eps_v6_avg_B * SCALING_FACTORS['eps'],
            sigma=0.3 * SCALING_FACTORS['eps'],
            lower=13.0 * SCALING_FACTORS['eps'],
            upper=16.0 * SCALING_FACTORS['eps'])
        
        # ------------------------------
        # 6. γ: Non-centered階層モデル（識別不能性解消 + 収束性改善）
        # ------------------------------
        # v7.1: Non-centered Parameterization（B形式も同様に適用）
        
        # ハイパーパラメータ（log空間で定義 → 正値保証）
        # B形式: 事前分布強化（識別不能性対策）
        log_gamma_mu_B = pm.Normal('log_gamma_mu',
            mu=np.log(GAMMA_HYPERPRIOR_MU),
            sigma=0.3)  # 0.5→0.3 より情報的に
        
        log_gamma_sd_B = pm.HalfNormal('log_gamma_sd', sigma=0.3)  # 0.5→0.3 より制約
        
        # Non-centered変換
        gamma_raw_B = pm.Normal('gamma_raw', mu=0, sigma=1, shape=7)
        
        gamma_vec_unscaled_B = pm.Deterministic('gamma_vec',
            pt.exp(log_gamma_mu_B + log_gamma_sd_B * gamma_raw_B))
        
        gamma_vec_scaled_B = pm.Deterministic('gamma_vec_scaled',
            pt.clip(gamma_vec_unscaled_B, 0.005, 0.5) * SCALING_FACTORS['gamma'])
        
        # 後方互換性のため個別gamma_i_scaledも定義
        for i in range(7):
            pm.Deterministic(f'gamma_{i+1}_scaled', gamma_vec_scaled_B[i])
        
        # 階層パラメータを物理値空間で記録（診断用）
        gamma_mean_scaled_B = pm.Deterministic('gamma_mean_scaled',
            pt.exp(log_gamma_mu_B) * SCALING_FACTORS['gamma'])
        gamma_std_scaled_B = pm.Deterministic('gamma_std_scaled',
            log_gamma_sd_B * SCALING_FACTORS['gamma'])
        
        # ------------------------------
        # 7. 尤度: StudentT（外れ値頑健）
        # ------------------------------
        model_op_B = ScaledInformedPriorModelOp(datasets, 'B')
        trans_pred_B = model_op_B(a_scale_scaled_B, gamma_vec_scaled_B,
                                   g_factor_scaled_B, B4_scaled_B,
                                   B6_scaled_B, eps_bg_scaled_B)
        
        likelihood_B = pm.StudentT('likelihood',
            nu=NU_STUDENTT,
            mu=trans_pred_B,
            sigma=sigma_eff,
            observed=trans_obs_concat)
        
        # ------------------------------
        # 8. SMCサンプリング（B形式専用強化設定）
        # ------------------------------
        print(f"\n🔬 SMCサンプリング開始（B形式 - 強化設定）")
        print(f"   Sampler: SMC (Sequential Monte Carlo)")
        print(f"   Draws: {SMC_DRAWS}, Chains: {SMC_CHAINS}, Cores: {SMC_CHAINS if SMC_PARALLEL else 1}")
        print(f"   ⚡ B形式強化: サンプル数2倍、チェーン数2倍、事前分布強化")
        print(f"   Hierarchical γ: ON, Likelihood: StudentT(ν={NU_STUDENTT})")
        
        trace_B = pm.sample_smc(
            draws=SMC_DRAWS,
            chains=SMC_CHAINS,
            cores=SMC_CHAINS if SMC_PARALLEL else 1,
            return_inferencedata=True,
            progressbar=True,
            random_seed=RANDOM_SEED
        )
        
        # SMCは自動的にlog_likelihoodを保存しないため、明示的に計算
        print(f"\n📊 log_likelihood計算中（B形式）...")
        pm.compute_log_likelihood(trace_B, model=model_B)
    
    print(f"\n✅ B形式サンプリング完了（log_likelihoodの保存済み）")
    
    # ============================================================================
    # LOO-CV評価
    # ============================================================================
    loo_H = compute_loo_cv(trace_H, 'H-form')
    loo_B = compute_loo_cv(trace_B, 'B-form')
    
    comparison_result = None
    if loo_H is not None and loo_B is not None:
        comparison_result = compare_models_loo(loo_H, loo_B)
    
    # ============================================================================
    # ベイズファクター計算
    # ============================================================================
    bf_result = compute_bayes_factor_smc(trace_H, trace_B)
    
    # ============================================================================
    # 可視化（修士論文用）
    # ============================================================================
    print(f"\n{'='*80}")
    print("📊 修士論文用プロット生成")
    print(f"{'='*80}")
    
    # H形式
    plot_prior_distributions(v6_params_H, 'H', results_dir)
    plot_posterior_distributions(trace_H, 'H', results_dir)
    plot_prior_posterior_comparison(trace_H, v6_params_H, 'H', results_dir)
    plot_posterior_predictive_spectra(trace_H, datasets, v6_params_H, 'H', results_dir)
    
    # B形式
    plot_prior_distributions(v6_params_B, 'B', results_dir)
    plot_posterior_distributions(trace_B, 'B', results_dir)
    plot_prior_posterior_comparison(trace_B, v6_params_B, 'B', results_dir)
    plot_posterior_predictive_spectra(trace_B, datasets, v6_params_B, 'B', results_dir)
    
    # H形式 vs B形式 比較プロット
    plot_posterior_predictive_spectra_comparison(trace_H, trace_B, datasets, results_dir)
    
    print(f"\n✅ 全プロット生成完了")
    
    # ============================================================================
    # 結果保存
    # ============================================================================
    print(f"\n{'='*80}")
    print("結果保存")
    print(f"{'='*80}")
    
    # トレース保存（SMCの場合、sample_statsにmixed typeがあるためpickle形式も用意）
    try:
        trace_H.to_netcdf(str(results_dir / 'trace_H.nc'))
        print("  ✓ trace_H.nc")
    except ValueError as e:
        print(f"  ⚠️ netCDF保存失敗 (SMC beta混在型): {e}")
        # pickle形式で保存
        import pickle
        with open(results_dir / 'trace_H.pkl', 'wb') as f:
            pickle.dump(trace_H, f)
        print("  ✓ trace_H.pkl (pickle形式)")
    
    try:
        trace_B.to_netcdf(str(results_dir / 'trace_B.nc'))
        print("  ✓ trace_B.nc")
    except ValueError as e:
        print(f"  ⚠️ netCDF保存失敗 (SMC beta混在型): {e}")
        import pickle
        with open(results_dir / 'trace_B.pkl', 'wb') as f:
            pickle.dump(trace_B, f)
        print("  ✓ trace_B.pkl (pickle形式)")
    
    # サマリー保存
    summary_H = az.summary(trace_H)
    summary_H.to_csv(results_dir / 'summary_H.csv')
    print("  ✓ summary_H.csv")
    
    summary_B = az.summary(trace_B)
    summary_B.to_csv(results_dir / 'summary_B.csv')
    print("  ✓ summary_B.csv")
    
    # 物理値への変換（スケーリング解除）- v7階層モデル対応
    posterior_H = trace_H.posterior
    params_H = {
        'g': float(posterior_H['g_factor_scaled'].mean()) / SCALING_FACTORS['g'],
        'a': float(posterior_H['a_scale_scaled'].mean()) / SCALING_FACTORS['a'],
        'B4': float(posterior_H['B4_scaled'].mean()) / SCALING_FACTORS['B4'],
        'B6': float(posterior_H['B6_scaled'].mean()) / SCALING_FACTORS['B6'],
        'eps': float(posterior_H['eps_bg_scaled'].mean()) / SCALING_FACTORS['eps'],
        'gamma': np.array([float(posterior_H[f'gamma_{i+1}_scaled'].mean()) / SCALING_FACTORS['gamma'] for i in range(7)]),
        # 階層パラメータ（v7新規）
        'gamma_mean': float(posterior_H['gamma_mean_scaled'].mean()) / SCALING_FACTORS['gamma'],
        'gamma_std': float(posterior_H['gamma_std_scaled'].mean()) / SCALING_FACTORS['gamma'],
    }
    
    posterior_B = trace_B.posterior
    params_B = {
        'g': float(posterior_B['g_factor_scaled'].mean()) / SCALING_FACTORS['g'],
        'a': float(posterior_B['a_scale_scaled'].mean()) / SCALING_FACTORS['a'],
        'B4': float(posterior_B['B4_scaled'].mean()) / SCALING_FACTORS['B4'],
        'B6': float(posterior_B['B6_scaled'].mean()) / SCALING_FACTORS['B6'],
        'eps': float(posterior_B['eps_bg_scaled'].mean()) / SCALING_FACTORS['eps'],
        'gamma': np.array([float(posterior_B[f'gamma_{i+1}_scaled'].mean()) / SCALING_FACTORS['gamma'] for i in range(7)]),
        # 階層パラメータ（v7新規）
        'gamma_mean': float(posterior_B['gamma_mean_scaled'].mean()) / SCALING_FACTORS['gamma'],
        'gamma_std': float(posterior_B['gamma_std_scaled'].mean()) / SCALING_FACTORS['gamma'],
    }
    
    # パラメータ保存
    params_H_df = pd.DataFrame([{k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in params_H.items()}])
    params_H_df.to_csv(results_dir / 'parameters_H.csv', index=False)
    print("  ✓ parameters_H.csv")
    
    params_B_df = pd.DataFrame([{k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in params_B.items()}])
    params_B_df.to_csv(results_dir / 'parameters_B.csv', index=False)
    print("  ✓ parameters_B.csv")
    
    # モデル評価結果保存（v7: SMC対応 + BF追加）
    if loo_H is not None and loo_B is not None:
        eval_results = {
            'H_form': loo_H,  # compute_model_evaluationの結果
            'B_form': loo_B,
            'comparison_waic': comparison_result if comparison_result else {},
            'comparison_bayes_factor': bf_result if bf_result else {},
            'timestamp': timestamp,
            'sampler': SAMPLER_TYPE,
            'likelihood': LIKELIHOOD_TYPE,
            'hierarchical_gamma': USE_HIERARCHICAL_GAMMA
        }
        
        with open(results_dir / 'model_evaluation.json', 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        print("  ✓ model_evaluation.json")
    
    # 完了
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("🎉 全処理完了")
    print(f"{'='*80}")
    print(f"  実行時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
    print(f"  結果: {results_dir}")
    print(f"  モデル形式: H-form & B-form (両方)")
    print(f"  サンプラー (H): {SAMPLER_TYPE} (Draws={SMC_DRAWS}, Chains={SMC_CHAINS})")
    print(f"  サンプラー (B): {SAMPLER_TYPE} (Draws={SMC_DRAWS}, Chains={SMC_CHAINS}) [強化]")
    print(f"  尤度: {LIKELIHOOD_TYPE} (ν={NU_STUDENTT})")
    print(f"  階層γモデル: {'ON' if USE_HIERARCHICAL_GAMMA else 'OFF'}")
    if comparison_result is not None:
        print(f"  推奨モデル (WAIC): {comparison_result.get('winner', 'N/A')}")
    if bf_result is not None:
        print(f"  推奨モデル (BayesFactor): {bf_result.get('winner', 'N/A')} (logBF={bf_result.get('log_BF', 0):.2f})")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
