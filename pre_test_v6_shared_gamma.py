"""
Global Fitting v6: Shared Gamma Model
全データセット共通の7-gammaモデル

【物理的根拠】
- γₖ: 準位|k⟩の固有緩和率（材料特性、温度・磁場に依存しない）
- 温度依存: Boltzmann分布で自動的に表現
- 磁場依存: Zeeman分裂で自動的に表現

【パラメータ数】
- v6: 12個 (5 global + 7 shared gamma) ← 84%削減

【期待効果】
- 条件数改善: 10¹⁶ → 10⁶-10⁸
- MCMC収束性: R-hat < 1.05, ESS > 400
- 物理的解釈: 明確（材料固有値）
"""

import os
import pathlib
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths
import unified_weighted_bayesian_fitting_final as uwbf
import warnings
from datetime import datetime
import json
from pathlib import Path
import traceback

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

# ==========================================
# ⚙️ 解析モデル設定
# ==========================================
MODEL_FORMS = ['B', 'H']

# 物理定数（診断用）
kB = 1.380649e-23  # Boltzmann定数 [J/K]

# パラメータスケーリング係数（条件数最適化版 v5）
# 目標: 最適化空間で全パラメータの幅を50程度に統一
# 境界拡張版: a=[0.1, 5.0], B₄=[0.1mK, 30mK], B₆=[-1mK, 1mK]
SCALING_FACTORS = {
    'g': 38.0,      # [1.5, 2.8] → [57, 106] (幅49)
    'a': 10.2,      # [0.1, 5.0] → [1.02, 51.0] (幅50) - 拡張
    'B4': 1672.0,   # [1e-4, 3e-2] → [0.17, 50.16] (幅50) - 拡張
    'B6': 25000.0,  # [-1e-3, 1e-3] → [-25, 25] (幅50) - 拡張
    'eps': 17.0,    # [13.0, 16.0] → [221, 272] (幅51)
    'gamma': 100.0  # [0.01, 0.5] → [1.0, 50.0] (幅49)
}

# データセット構成
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

# ==========================================
# 📊 パラメータ管理
# ==========================================
def detect_polariton_modes(freq, trans, polariton_upper_limit=0.361505):
    """
    透過スペクトルからポラリトンモードを検出
    
    Parameters:
    -----------
    freq : array
        周波数配列 [THz]
    trans : array
        透過率配列
    polariton_upper_limit : float
        ポラリトンモード領域の周波数上限 [THz] = 0.361505
    
    Returns:
    --------
    has_polariton : bool
        ポラリトンモード(UP/LP)が検出されたか
    """
    from scipy.signal import find_peaks
    
    # ピーク検出（透過率が高い部分）
    peaks, _ = find_peaks(trans, prominence=0.05, width=3)
    
    if len(peaks) == 0:
        return False
    
    peak_freqs = freq[peaks]
    
    # ポラリトンモード領域のピークをカウント
    polariton_peaks = peak_freqs[peak_freqs <= polariton_upper_limit]
    
    # 2個以上の低周波ピーク = ポラリトンモード形成（UP/LP）
    return len(polariton_peaks) >= 2

def detect_peaks_and_classify(freq, trans, polariton_upper_limit=0.361505, cavity_lower_limit=0.45):
    """
    ピークを検出し、ポラリトンモード vs 共振器モードに分類
    透過率が高い部分（透過スペクトルのピーク）を検出
    """
    from scipy.signal import find_peaks
    
    # 透過率の極大値を検出
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
        
        if pf <= polariton_upper_limit:
            f_end_clipped = min(f_end, polariton_upper_limit)
            if f_end_clipped > f_start:
                polariton_regions.append((f_start, f_end_clipped))
        elif pf >= cavity_lower_limit:
            f_start_clipped = max(f_start, cavity_lower_limit)
            if f_end > f_start_clipped:
                cavity_regions.append((f_start_clipped, f_end))
    
    return polariton_regions, cavity_regions

def create_weight_array(freq, trans, polariton_regions, cavity_regions):
    """重み配列生成: ポラリトン=1.5, 共振器=1.0, それ以外=0.01"""
    weight_array = np.full_like(freq, 0.01)  # デフォルト: 0.01
    
    # ポラリトン領域: 1.5
    for f_start, f_end in polariton_regions:
        mask = (freq >= f_start) & (freq <= f_end)
        weight_array[mask] = 1.5
    
    # 共振器領域: 1.0
    for f_start, f_end in cavity_regions:
        mask = (freq >= f_start) & (freq <= f_end)
        weight_array[mask] = 1.0
    
    return weight_array

def load_all_datasets(target_data_list):
    """複数データセット読み込み"""
    print("\n--- データ読み込み ---")
    
    datasets = []
    base_dir = Path(__file__).parent / 'bayesian_inputs'
    
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
                'weight_array': weight_array,  # 周波数ごとの重み配列
                'B': config['B'],
                'T': config['T'],
                'label': label,
                'polariton_regions': polariton_regions,
                'cavity_regions': cavity_regions,
                'sigma': 0.01  # 基本ノイズレベル
            }
            
            datasets.append(dataset)
            
            print(f"✓ {label} (B={config['B']}T, T={config['T']}K): {len(freq)} points")
            print(f"  Polariton領域 (1.5×): {len(polariton_regions)} regions")
            print(f"  Cavity領域 (1.0×): {len(cavity_regions)} regions")
            
        except Exception as e:
            print(f"❌ {config['col']} 読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(datasets) == 0:
        print("❌ データセットが1つも読み込めませんでした")
    else:
        print(f"\n✅ 合計 {len(datasets)} データセット読み込み完了")
    
    return datasets

def pack_shared_gamma_parameters(global_dict, gamma_shared):
    """
    共有gammaパラメータを1次元配列に変換
    
    Parameters
    ----------
    global_dict : dict
        {'g', 'a', 'B4', 'B6', 'eps'}
    gamma_shared : np.ndarray
        [7個] 全データセットで共通
    
    Returns
    -------
    params_flat : np.ndarray
        [g, a, B4, B6, eps, γ₁, γ₂, ..., γ₇]
        合計：5 + 7 = 12個
    """
    params_flat = [
        global_dict['g'] * SCALING_FACTORS['g'],
        global_dict['a'] * SCALING_FACTORS['a'],
        global_dict['B4'] * SCALING_FACTORS['B4'],
        global_dict['B6'] * SCALING_FACTORS['B6'],
        global_dict['eps'] * SCALING_FACTORS['eps']
    ]
    
    for g in gamma_shared:
        params_flat.append(g * SCALING_FACTORS['gamma'])
    
    return np.array(params_flat)

def unpack_shared_gamma_parameters(params_flat):
    """1次元配列を辞書に分解"""
    global_scaled = {
        'g': params_flat[0],
        'a': params_flat[1],
        'B4': params_flat[2],
        'B6': params_flat[3],
        'eps': params_flat[4]
    }
    
    gamma_shared = params_flat[5:12]  # 7個
    
    return global_scaled, gamma_shared

# ==========================================
# 🎯 共有Gamma Residuals
# ==========================================
def shared_gamma_residuals(params_flat, datasets, model_form='H'):
    """全データセットで同じgammaを使用した残差計算"""
    global_scaled, gamma_shared = unpack_shared_gamma_parameters(params_flat)
    
    # 物理単位への復元
    g = global_scaled['g'] / SCALING_FACTORS['g']
    a_scale = global_scaled['a'] / SCALING_FACTORS['a']
    B4 = global_scaled['B4'] / SCALING_FACTORS['B4']
    B6 = global_scaled['B6'] / SCALING_FACTORS['B6']
    eps_bg = global_scaled['eps'] / SCALING_FACTORS['eps']
    gamma_array = gamma_shared / SCALING_FACTORS['gamma']
    
    # 物理的範囲に制限
    gamma_array = np.clip(gamma_array, 0.005, 0.4)
    
    residuals = []
    N_spin = 1.9386e+28
    d_fixed = 157.8e-6
    
    for data in datasets:
        # ハミルトニアン（データセット固有のB, T）
        H = uwbf.get_hamiltonian(data['B'], g, B4, B6)
        
        # 感受率（共通のgamma使用）
        chi_raw = uwbf.calculate_susceptibility(data['freq'], H, data['T'], gamma_array)
        
        # スケーリング
        G0 = a_scale * uwbf.mu0 * N_spin * (g * uwbf.muB)**2 / (2 * uwbf.hbar) / uwbf.THZ_TO_RAD_S
        chi = G0 * chi_raw
        
        # 比透磁率
        if model_form == 'H':
            mu_r = 1.0 + chi
        elif model_form == 'B':
            denominator = 1.0 - chi
            mu_r = 1.0 / denominator
        
        # 透過率
        trans = uwbf.calculate_transmission(data['freq'], mu_r, d_fixed, eps_bg)
        
        # NaNチェック
        if np.any(~np.isfinite(trans)):
            trans = np.nan_to_num(trans, nan=0.5)
        
        # 周波数ごとの重み付き残差
        # weight_array: ポラリトン領域=1.5, それ以外=1.0
        effective_sigma = data['sigma'] / np.sqrt(data['weight_array'])
        res = (data['trans'] - trans) / effective_sigma
        residuals.append(res)
    
    return np.concatenate(residuals)

# ==========================================
# 🚀 初期値・境界値
# ==========================================
def generate_shared_gamma_initial_values():
    """共有gammaモデルの初期値"""
    print("\n🔧 Generating shared gamma initial values...")
    
    # Global parameters
    global_phys = {
        'g': 1.95,
        'a': 1.0,      
        'B4': 2.02*1.0e-3, # 山田の論文値参考
        'B6': -1.2*1.0e-5, # 山田の論文値参考 
        'eps': 14.4        # Elijahらの実験値参考
    }
    
    # 共有gamma（材料固有値の推定）
    # pre_test_v4の結果から典型的な値を抽出
    gamma_shared = np.array([0.10, 0.15, 0.12, 0.11, 0.14, 0.13, 0.16])
    
    return global_phys, gamma_shared

def get_shared_gamma_bounds():
    """共有gammaモデルの境界値
    
    【パラメータ範囲の物理的根拠】
    
    B₄, B₆ (結晶場パラメータ):
        - Gd³⁺イオン (4f⁷, S=7/2) の希土類ガーネット結晶で一般的な範囲を採用
        - B₄: 0.5 mK ～ 20 mK (典型値: 1-5 mK)
        - B₆: ±0.5 mK (B₄より1-2桁小さい)
        - 注: 山田の論文値 (B₄=2.02mK, B₆=-1.2×10⁻⁵K) は参考値として使用
    
    a_scale (結合定数スケーリング):
        - G₀ = a × (μ₀ N_spin (g μ_B)²) / (2ℏ) / THz_TO_RAD_S
        - 理論的には a = 1.0 だが、以下の不確定性を考慮:
          * サンプル厚さの誤差 (±20-50%)
          * スピン密度 N_spin の不確定性
          * 光学定数の補正
        - 拡張範囲: 0.1 ～ 5.0 (実験誤差±5倍を許容)
    """
    # Global parameters
    g_min, g_max = 1.5, 2.8           # g因子: Gd³⁺の一般値 ～2.0 (維持)
    a_min, a_max = 0.1, 5.0           # 拡張: [0.3, 5.0] → [0.1, 5.0] 実験誤差対応
    B4_min, B4_max = 1.0e-4, 3.0e-2   # 拡張: Gd³⁺ガーネット (0.1-30 mK) H/B形式対応
    B6_min, B6_max = -1.0e-3, 1.0e-3  # 拡張: [-0.5mK, 0.5mK] → [-1mK, 1mK]
    eps_min, eps_max = 13.0, 16.0     # 誘電率: GGG一般値 (維持)
    
    # Shared gamma (維持: 現在の設定で物理的妥当性あり)
    gamma_min, gamma_max = 0.01, 0.5
    
    lower = [
        g_min * SCALING_FACTORS['g'],
        a_min * SCALING_FACTORS['a'],
        B4_min * SCALING_FACTORS['B4'],
        B6_min * SCALING_FACTORS['B6'],
        eps_min * SCALING_FACTORS['eps']
    ]
    upper = [
        g_max * SCALING_FACTORS['g'],
        a_max * SCALING_FACTORS['a'],
        B4_max * SCALING_FACTORS['B4'],
        B6_max * SCALING_FACTORS['B6'],
        eps_max * SCALING_FACTORS['eps']
    ]
    
    # 7個の共有gamma
    lower.extend([gamma_min * SCALING_FACTORS['gamma']] * 7)
    upper.extend([gamma_max * SCALING_FACTORS['gamma']] * 7)
    
    return np.array(lower), np.array(upper)

# ==========================================
# 📊 フィット品質解析
# ==========================================
def analyze_shared_gamma_fit_quality(datasets, params_flat, model_form):
    """フィット品質の統計分析"""
    global_scaled, gamma_shared = unpack_shared_gamma_parameters(params_flat)
    
    # 物理値
    g = global_scaled['g'] / SCALING_FACTORS['g']
    a_scale = global_scaled['a'] / SCALING_FACTORS['a']
    B4 = global_scaled['B4'] / SCALING_FACTORS['B4']
    B6 = global_scaled['B6'] / SCALING_FACTORS['B6']
    eps_bg = global_scaled['eps'] / SCALING_FACTORS['eps']
    gamma_array = gamma_shared / SCALING_FACTORS['gamma']
    gamma_array = np.clip(gamma_array, 0.01, 0.5)
    
    stats = []
    N_spin = 1.9386e+28
    d_fixed = 157.8e-6
    
    for data in datasets:
        H = uwbf.get_hamiltonian(data['B'], g, B4, B6)
        chi_raw = uwbf.calculate_susceptibility(data['freq'], H, data['T'], gamma_array)
        G0 = a_scale * uwbf.mu0 * N_spin * (g * uwbf.muB)**2 / (2 * uwbf.hbar) / uwbf.THZ_TO_RAD_S
        chi = G0 * chi_raw
        
        if model_form == 'H':
            mu_r = 1.0 + chi
        elif model_form == 'B':
            denominator = 1.0 - chi
            mu_r = 1.0 / denominator
        
        trans = uwbf.calculate_transmission(data['freq'], mu_r, d_fixed, eps_bg)
        trans = np.nan_to_num(trans, nan=0.5)
        
        residuals = data['trans'] - trans
        rmse = np.sqrt(np.mean(residuals**2))
        max_error = np.max(np.abs(residuals))
        r_squared = 1 - np.sum(residuals**2) / np.sum((data['trans'] - np.mean(data['trans']))**2)
        
        chi_abs = np.abs(chi)
        unstable_fraction = 0.0
        if model_form == 'B':
            unstable_mask = chi_abs > 0.9
            unstable_fraction = np.sum(unstable_mask) / len(chi_abs) * 100
        
        stats.append({
            'label': data['label'],
            'B': data['B'],
            'T': data['T'],
            'rmse': rmse,
            'max_error': max_error,
            'r_squared': r_squared,
            'chi_max': np.max(chi_abs),
            'chi_mean': np.mean(chi_abs),
            'unstable_%': unstable_fraction
        })
    
    return pd.DataFrame(stats)

# ==========================================
# 🔍 Diagnostic Functions
# ==========================================
def diagnose_problematic_regions(datasets, params_flat, model_form):
    """
    失敗領域の物理的診断
    - Boltzmann分布（基底状態占有率）
    - エネルギーギャップ
    - χの最大値
    - 数値安定性の評価
    """
    global_scaled, gamma_shared = unpack_shared_gamma_parameters(params_flat)
    
    # 物理値への復元
    g = global_scaled['g'] / SCALING_FACTORS['g']
    a_scale = global_scaled['a'] / SCALING_FACTORS['a']
    B4 = global_scaled['B4'] / SCALING_FACTORS['B4']
    B6 = global_scaled['B6'] / SCALING_FACTORS['B6']
    eps_bg = global_scaled['eps'] / SCALING_FACTORS['eps']
    gamma_array = gamma_shared / SCALING_FACTORS['gamma']
    gamma_array = np.clip(gamma_array, 0.005, 0.5)
    
    N_spin = 1.9386e+28
    d_fixed = 157.8e-6
    
    print("\n" + "="*80)
    print("🔍 Physical Diagnosis of Problematic Regions")
    print("="*80)
    
    diagnostics = []
    
    for data in datasets:
        # ハミルトニアン固有値計算
        H = uwbf.get_hamiltonian(data['B'], g, B4, B6)
        E_vals, U = np.linalg.eigh(H)
        
        # Boltzmann分布
        kT = kB * data['T']  # [J]
        E_vals_J = E_vals * uwbf.hbar * uwbf.THZ_TO_RAD_S  # [J]に変換
        Z = np.sum(np.exp(-E_vals_J / kT))
        pops = np.exp(-E_vals_J / kT) / Z
        
        # エネルギーギャップ（meV）
        energy_gap_meV = (E_vals[1] - E_vals[0]) * uwbf.hbar * uwbf.THZ_TO_RAD_S * 1000 / 1.602176634e-19
        
        # 感受率計算
        chi_raw = uwbf.calculate_susceptibility(data['freq'], H, data['T'], gamma_array)
        G0 = a_scale * uwbf.mu0 * N_spin * (g * uwbf.muB)**2 / (2 * uwbf.hbar) / uwbf.THZ_TO_RAD_S
        chi = G0 * chi_raw
        chi_abs = np.abs(chi)
        chi_max = np.max(chi_abs)
        chi_mean = np.mean(chi_abs)
        
        # B-form安定性
        if model_form == 'B':
            denominator = 1.0 - chi
        else:
            unstable_fraction = 0.0
        
        # フィッティング品質
        if model_form == 'H':
            mu_r = 1.0 + chi
        elif model_form == 'B':
            denominator = 1.0 - chi
            mu_r = 1.0 / denominator
        
        trans_fit = uwbf.calculate_transmission(data['freq'], mu_r, d_fixed, eps_bg)
        trans_fit = np.nan_to_num(trans_fit, nan=0.5)
        
        residuals = data['trans'] - trans_fit
        rmse = np.sqrt(np.mean(residuals**2))
        max_error = np.max(np.abs(residuals))
        r_squared = 1 - np.sum(residuals**2) / np.sum((data['trans'] - np.mean(data['trans']))**2)
        
        chi_abs = np.abs(chi)
        unstable_fraction = 0.0
        if model_form == 'B':
            unstable_mask = chi_abs > 0.9
            unstable_fraction = np.sum(unstable_mask) / len(chi_abs) * 100
        
        # 診断フラグ
        warnings = []
        if pops[0] > 0.95:
            warnings.append("基底状態支配的(>95%)")
        if chi_max > 1.0 and model_form == 'B':
            warnings.append("B-form数値不安定(|χ|>1)")
        if chi_max > 0.8 and model_form == 'H':
            warnings.append("H-form |χ|高(>0.8)")
        if rmse > 0.15:
            warnings.append(f"フィット失敗(RMSE={rmse:.3f})")
        
        status = "⚠️ 問題あり" if warnings else "✅ 正常"
        
        diagnostics.append({
            'label': data['label'],
            'B': data['B'],
            'T': data['T'],
            'pop_ground': pops[0],
            'pop_1st_excited': pops[1],
            'energy_gap_meV': energy_gap_meV,
            'chi_max': chi_max,
            'chi_mean': chi_mean,
            'rmse': rmse,
            'status': status,
            'warnings': '; '.join(warnings) if warnings else 'None'
        })
        
        print(f"\n{data['label']} (B={data['B']}T, T={data['T']}K) {status}")
        print(f"  基底状態占有率: {pops[0]:.4f} (第1励起: {pops[1]:.4f})")
        print(f"  エネルギーギャップ (E₁-E₀): {energy_gap_meV:.3f} meV (vs kT={data['T']*0.0862:.2f} meV)")
        print(f"  Max|χ|: {chi_max:.3f}, Mean|χ|: {chi_mean:.3f}")
        print(f"  RMSE: {rmse:.4f}")
        if warnings:
            print(f"  ⚠️ 警告: {'; '.join(warnings)}")
    
    # 統計サマリー
    print("\n" + "="*80)
    print("📊 Diagnostic Summary")
    print("="*80)
    
    df_diag = pd.DataFrame(diagnostics)
    
    # 問題のあるデータセットを抽出
    problematic = df_diag[df_diag['status'].str.contains('問題')]
    if len(problematic) > 0:
        print(f"\n⚠️ 問題のあるデータセット: {len(problematic)}/{len(df_diag)}")
        print(problematic[['label', 'B', 'T', 'pop_ground', 'chi_max', 'rmse', 'warnings']].to_string(index=False))
        
        # 除外推奨の判定
        severe = df_diag[df_diag['rmse'] > 0.15]
        if len(severe) > 0:
            print(f"\n🔴 除外推奨（RMSE > 0.15）: {len(severe)}件")
            for _, row in severe.iterrows():
                print(f"  - {row['label']}: RMSE={row['rmse']:.3f}, |χ|_max={row['chi_max']:.2f}")
    else:
        print("\n✅ 全データセット正常")
    
    # 物理的パターンの検出
    print("\n" + "-"*80)
    print("🔬 Physical Pattern Analysis")
    print("-"*80)
    
    # 低温領域
    low_temp = df_diag[df_diag['T'] <= 10]
    if len(low_temp) > 0:
        print(f"\n低温領域 (T ≤ 10K): {len(low_temp)}件")
        print(f"  平均基底占有率: {low_temp['pop_ground'].mean():.3f}")
        print(f"  平均RMSE: {low_temp['rmse'].mean():.4f}")
        print(f"  平均|χ|_max: {low_temp['chi_max'].mean():.3f}")
    
    # 高磁場領域
    high_field = df_diag[df_diag['B'] >= 8.0]
    if len(high_field) > 0:
        print(f"\n高磁場領域 (B ≥ 8T): {len(high_field)}件")
        print(f"  平均RMSE: {high_field['rmse'].mean():.4f}")
        print(f"  平均|χ|_max: {high_field['chi_max'].mean():.3f}")
    
    # 極端条件（低温+高磁場）
    extreme = df_diag[(df_diag['T'] <= 10) & (df_diag['B'] >= 8.0)]
    if len(extreme) > 0:
        print(f"\n極端条件 (T ≤ 10K & B ≥ 8T): {len(extreme)}件")
        print(f"  平均RMSE: {extreme['rmse'].mean():.4f}")
        if extreme['rmse'].mean() > 0.15:
            print("  ⚠️ この領域は単イオン近似の適用範囲外の可能性")
    
    print("\n" + "="*80)
    
    return df_diag

# ==========================================
# 📊 Plotting Functions
# ==========================================
def plot_all_fits(datasets, params_flat, output_dir, model_form):
    """全スペクトルのフィット結果をプロット（重み付け領域を水色で強調）"""
    global_scaled, gamma_shared = unpack_shared_gamma_parameters(params_flat)
    
    # 物理値への復元
    g = global_scaled['g'] / SCALING_FACTORS['g']
    a_scale = global_scaled['a'] / SCALING_FACTORS['a']
    B4 = global_scaled['B4'] / SCALING_FACTORS['B4']
    B6 = global_scaled['B6'] / SCALING_FACTORS['B6']
    eps_bg = global_scaled['eps'] / SCALING_FACTORS['eps']
    gamma_array = gamma_shared / SCALING_FACTORS['gamma']
    gamma_array = np.clip(gamma_array, 0.005, 0.5)
    
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    N_spin = 1.9386e+28
    d_fixed = 157.8e-6
    
    for i, (data, ax) in enumerate(zip(datasets, axes)):
        # フィットスペクトル計算
        H = uwbf.get_hamiltonian(data['B'], g, B4, B6)
        chi_raw = uwbf.calculate_susceptibility(data['freq'], H, data['T'], gamma_array)
        G0 = a_scale * uwbf.mu0 * N_spin * (g * uwbf.muB)**2 / (2 * uwbf.hbar) / uwbf.THZ_TO_RAD_S
        chi = G0 * chi_raw
        
        if model_form == 'H':
            mu_r = 1.0 + chi
        elif model_form == 'B':
            denominator = 1.0 - chi
            mu_r = 1.0 / denominator
        
        y_fit = uwbf.calculate_transmission(data['freq'], mu_r, d_fixed, eps_bg)
        y_fit = np.nan_to_num(y_fit, nan=0.5)
        
        # ポラリトンモードと共振器モードを検出して色分け
        polariton_regions, cavity_regions = detect_peaks_and_classify(data['freq'], y_fit)
        
        # ポラリトン領域（1.5×重み）を赤で塗りつぶし
        polariton_legend_added = False
        for freq_start, freq_end in polariton_regions:
            label = 'Polariton (1.5×)' if not polariton_legend_added else None
            ax.axvspan(freq_start, freq_end, alpha=0.15, color='orange', label=label, zorder=1)
            polariton_legend_added = True
        
        # 共振器領域（1.0×重み）を水色で塗りつぶし
        cavity_legend_added = False
        for freq_start, freq_end in cavity_regions:
            label = 'Cavity (1.0×)' if not cavity_legend_added else None
            ax.axvspan(freq_start, freq_end, alpha=0.15, color='green', label=label, zorder=1)
            cavity_legend_added = True
        
        # データとフィット（重み付け領域の上に描画）
        ax.plot(data['freq'], data['trans'], 'o', color='gray', 
                markersize=2.5, alpha=0.6, label='Data', zorder=2)
        ax.plot(data['freq'], y_fit, 'r-', linewidth=2.0, label='Fit', zorder=3)
        
        # 残差の表示
        residuals = data['trans'] - y_fit
        rmse = np.sqrt(np.mean(residuals**2))
        
        ax.set_title(f"{data['label']} (RMSE: {rmse:.4f})", fontsize=11, fontweight='bold')
        ax.set_xlabel('Frequency (THz)', fontsize=10)
        ax.set_ylabel('Transmittance', fontsize=10)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
        
        # y軸範囲の自動調整
        y_margin = 0.05
        y_min = min(np.min(data['trans']), np.min(y_fit)) - y_margin
        y_max = max(np.max(data['trans']), np.max(y_fit)) + y_margin
        ax.set_ylim(y_min, y_max)
    
    # 未使用の軸を非表示
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fit_all_spectra.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fit_all_spectra.png saved")

def plot_all_fits_comparison(datasets, params_H, params_B, output_dir):
    """H形式とB形式を1枚のグラフに重ねてプロット（H:赤, B:青）"""
    # H形式パラメータ
    global_H, gamma_H = unpack_shared_gamma_parameters(params_H)
    g_H = global_H['g'] / SCALING_FACTORS['g']
    a_H = global_H['a'] / SCALING_FACTORS['a']
    B4_H = global_H['B4'] / SCALING_FACTORS['B4']
    B6_H = global_H['B6'] / SCALING_FACTORS['B6']
    eps_H = global_H['eps'] / SCALING_FACTORS['eps']
    gamma_H_array = np.clip(gamma_H / SCALING_FACTORS['gamma'], 0.005, 0.5)
    
    # B形式パラメータ
    global_B, gamma_B = unpack_shared_gamma_parameters(params_B)
    g_B = global_B['g'] / SCALING_FACTORS['g']
    a_B = global_B['a'] / SCALING_FACTORS['a']
    B4_B = global_B['B4'] / SCALING_FACTORS['B4']
    B6_B = global_B['B6'] / SCALING_FACTORS['B6']
    eps_B = global_B['eps'] / SCALING_FACTORS['eps']
    gamma_B_array = np.clip(gamma_B / SCALING_FACTORS['gamma'], 0.005, 0.5)
    
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    N_spin = 1.9386e+28
    d_fixed = 157.8e-6
    
    for i, (data, ax) in enumerate(zip(datasets, axes)):
        # H形式フィット計算
        H_ham_H = uwbf.get_hamiltonian(data['B'], g_H, B4_H, B6_H)
        chi_raw_H = uwbf.calculate_susceptibility(data['freq'], H_ham_H, data['T'], gamma_H_array)
        G0_H = a_H * uwbf.mu0 * N_spin * (g_H * uwbf.muB)**2 / (2 * uwbf.hbar) / uwbf.THZ_TO_RAD_S
        chi_H = G0_H * chi_raw_H
        mu_r_H = 1.0 + chi_H
        trans_H = uwbf.calculate_transmission(data['freq'], mu_r_H, d_fixed, eps_H)
        trans_H = np.nan_to_num(trans_H, nan=0.5)
        
        # B形式フィット計算
        H_ham_B = uwbf.get_hamiltonian(data['B'], g_B, B4_B, B6_B)
        chi_raw_B = uwbf.calculate_susceptibility(data['freq'], H_ham_B, data['T'], gamma_B_array)
        G0_B = a_B * uwbf.mu0 * N_spin * (g_B * uwbf.muB)**2 / (2 * uwbf.hbar) / uwbf.THZ_TO_RAD_S
        chi_B = G0_B * chi_raw_B
        denominator_B = 1.0 - chi_B
        mu_r_B = 1.0 / denominator_B
        trans_B = uwbf.calculate_transmission(data['freq'], mu_r_B, d_fixed, eps_B)
        trans_B = np.nan_to_num(trans_B, nan=0.5)
        
        # ポラリトン/共振器領域の検出（H形式基準）
        polariton_regions, cavity_regions = detect_peaks_and_classify(data['freq'], trans_H)
        
        # 領域の塗りつぶし
        polariton_legend_added = False
        for freq_start, freq_end in polariton_regions:
            label = 'Polariton (1.5×)' if not polariton_legend_added else None
            ax.axvspan(freq_start, freq_end, alpha=0.12, color='orange', label=label, zorder=1)
            polariton_legend_added = True
        
        cavity_legend_added = False
        for freq_start, freq_end in cavity_regions:
            label = 'Cavity (1.0×)' if not cavity_legend_added else None
            ax.axvspan(freq_start, freq_end, alpha=0.12, color='green', label=label, zorder=1)
            cavity_legend_added = True
        
        # データとフィット結果をプロット
        ax.plot(data['freq'], data['trans'], 'o', color='gray', 
                markersize=2.5, alpha=0.6, label='Data', zorder=2)
        ax.plot(data['freq'], trans_H, '-', color='red', linewidth=2.0, 
                label='H-form', zorder=3)
        ax.plot(data['freq'], trans_B, '-', color='blue', linewidth=2.0, 
                label='B-form', zorder=3)
        
        # RMSE計算
        rmse_H = np.sqrt(np.mean((data['trans'] - trans_H)**2))
        rmse_B = np.sqrt(np.mean((data['trans'] - trans_B)**2))
        
        ax.set_title(f"{data['label']}\nH-RMSE: {rmse_H:.4f}, B-RMSE: {rmse_B:.4f}", 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (THz)', fontsize=10)
        ax.set_ylabel('Transmittance', fontsize=10)
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
        
        # y軸範囲の自動調整
        y_margin = 0.05
        y_min = min(np.min(data['trans']), np.min(trans_H), np.min(trans_B)) - y_margin
        y_max = max(np.max(data['trans']), np.max(trans_H), np.max(trans_B)) + y_margin
        ax.set_ylim(y_min, y_max)
    
    # 未使用の軸を非表示
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fit_all_spectra_HB_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fit_all_spectra_HB_comparison.png saved")

def plot_residuals(datasets, params_flat, output_dir, model_form):
    """残差プロット（系統誤差の検出用）"""
    global_scaled, gamma_shared = unpack_shared_gamma_parameters(params_flat)
    
    # 物理値への復元
    g = global_scaled['g'] / SCALING_FACTORS['g']
    a_scale = global_scaled['a'] / SCALING_FACTORS['a']
    B4 = global_scaled['B4'] / SCALING_FACTORS['B4']
    B6 = global_scaled['B6'] / SCALING_FACTORS['B6']
    eps_bg = global_scaled['eps'] / SCALING_FACTORS['eps']
    gamma_array = gamma_shared / SCALING_FACTORS['gamma']
    gamma_array = np.clip(gamma_array, 0.005, 0.4)
    
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    N_spin = 1.9386e+28
    d_fixed = 157.8e-6
    
    for i, (data, ax) in enumerate(zip(datasets, axes)):
        # フィットスペクトル計算
        H = uwbf.get_hamiltonian(data['B'], g, B4, B6)
        chi_raw = uwbf.calculate_susceptibility(data['freq'], H, data['T'], gamma_array)
        G0 = a_scale * uwbf.mu0 * N_spin * (g * uwbf.muB)**2 / (2 * uwbf.hbar) / uwbf.THZ_TO_RAD_S
        chi = G0 * chi_raw
        
        if model_form == 'H':
            mu_r = 1.0 + chi
        elif model_form == 'B':
            denominator = 1.0 - chi
            mu_r = 1.0 / denominator
        
        y_fit = uwbf.calculate_transmission(data['freq'], mu_r, d_fixed, eps_bg)
        y_fit = np.nan_to_num(y_fit, nan=0.5)
        
        residuals = data['trans'] - y_fit
        
        # ポラリトンモードと共振器モードを検出して色分け
        polariton_regions, cavity_regions = detect_peaks_and_classify(data['freq'], y_fit)
        
        # ポラリトン領域（1.5×重み）をオレンジで塗りつぶし
        polariton_legend_added = False
        for freq_start, freq_end in polariton_regions:
            label = 'Polariton (1.5×)' if not polariton_legend_added else None
            ax.axvspan(freq_start, freq_end, alpha=0.15, color='orange', label=label, zorder=1)
            polariton_legend_added = True
        
        # 共振器領域（1.0×重み）を緑で塗りつぶし
        cavity_legend_added = False
        for freq_start, freq_end in cavity_regions:
            label = 'Cavity (1.0×)' if not cavity_legend_added else None
            ax.axvspan(freq_start, freq_end, alpha=0.15, color='green', label=label, zorder=1)
            cavity_legend_added = True
        
        # 残差プロット
        ax.plot(data['freq'], residuals, 'o-', color='steelblue', 
                markersize=3, linewidth=1, alpha=0.7, zorder=2)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, 
                  label='Zero Line', zorder=1)
        
        # 統計情報
        rmse = np.sqrt(np.mean(residuals**2))
        mean_res = np.mean(residuals)
        
        ax.set_title(f"{data['label']} (RMSE: {rmse:.4f}, Mean: {mean_res:.4f})", 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Frequency (THz)', fontsize=10)
        ax.set_ylabel('Residuals', fontsize=10)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
    
    # 未使用の軸を非表示
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_all_spectra.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ residuals_all_spectra.png saved")

def plot_chi_distribution(datasets, params_flat, output_dir, model_form):
    """χ分布の可視化（H-form vs B-form診断用）"""
    global_scaled, gamma_shared = unpack_shared_gamma_parameters(params_flat)
    
    # 物理値への復元
    g = global_scaled['g'] / SCALING_FACTORS['g']
    a_scale = global_scaled['a'] / SCALING_FACTORS['a']
    B4 = global_scaled['B4'] / SCALING_FACTORS['B4']
    B6 = global_scaled['B6'] / SCALING_FACTORS['B6']
    eps_bg = global_scaled['eps'] / SCALING_FACTORS['eps']
    gamma_array = gamma_shared / SCALING_FACTORS['gamma']
    gamma_array = np.clip(gamma_array, 0.005, 0.4)
    
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    N_spin = 1.9386e+28
    
    for i, (data, ax) in enumerate(zip(datasets, axes)):
        # χ計算
        H = uwbf.get_hamiltonian(data['B'], g, B4, B6)
        chi_raw = uwbf.calculate_susceptibility(data['freq'], H, data['T'], gamma_array)
        G0 = a_scale * uwbf.mu0 * N_spin * (g * uwbf.muB)**2 / (2 * uwbf.hbar) / uwbf.THZ_TO_RAD_S
        chi = G0 * chi_raw
        
        # 実部と虚部をプロット
        chi_real = np.real(chi)
        chi_imag = np.imag(chi)
        
        ax.plot(data['freq'], chi_real, '-', color='blue', linewidth=2.0, label="Re(χ)")
        ax.plot(data['freq'], chi_imag, '--', color='red', linewidth=2.0, label="Im(χ)")
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)  # ゼロ基準線
        
        # B-formの危険領域（Re(χ) > 0.9）を強調
        if model_form == 'B':
            danger_mask = chi_real > 0.9
            if np.any(danger_mask):
                ax.axhspan(0.9, ax.get_ylim()[1], alpha=0.15, color='red', 
                          label='Danger Zone (Re(χ)>0.9)', zorder=1)
                ax.axhline(0.9, color='orange', linestyle='--', linewidth=1.5, 
                          label='Warning Threshold', alpha=0.7)
        
        # 統計情報
        chi_real_max = np.max(np.abs(chi_real))
        chi_imag_max = np.max(np.abs(chi_imag))
        
        ax.set_title(f"{data['label']}\nMax|Re(χ)|={chi_real_max:.3f}, Max|Im(χ)|={chi_imag_max:.3f}", 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (THz)', fontsize=10)
        ax.set_ylabel('χ (Magnetic Susceptibility)', fontsize=10)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        # y軸範囲の自動調整
        y_margin = max(chi_real_max, chi_imag_max) * 0.1
        y_max = max(chi_real_max, chi_imag_max) * 1.1
        y_min = min(np.min(chi_real), np.min(chi_imag)) - y_margin
        ax.set_ylim(y_min, y_max)
    
    # 未使用の軸を非表示
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chi_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ chi_distribution.png saved")

def plot_energy_levels_and_populations(datasets, params_flat, output_dir, model_form):
    """エネルギー固有値と占有確率のプロット（8準位表示）"""
    global_scaled, gamma_shared = unpack_shared_gamma_parameters(params_flat)
    
    # 物理値への復元
    g = global_scaled['g'] / SCALING_FACTORS['g']
    a_scale = global_scaled['a'] / SCALING_FACTORS['a']
    B4 = global_scaled['B4'] / SCALING_FACTORS['B4']
    B6 = global_scaled['B6'] / SCALING_FACTORS['B6']
    eps_bg = global_scaled['eps'] / SCALING_FACTORS['eps']
    
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))
    
    # エネルギー準位プロット
    fig_energy, axes_e = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes_e = axes_e.flatten() if n_datasets > 1 else [axes_e]
    
    for i, (data, ax) in enumerate(zip(datasets, axes_e)):
        # ハミルトニアン構築
        H_ham = uwbf.get_hamiltonian(data['B'], g, B4, B6)
        E_vals, _ = np.linalg.eigh(H_ham)
        
        # 絶対的なエネルギー固有値をmeVに変換（基準値を引かない）
        E_vals_meV = E_vals * 0.0862  # K → meV
        
        x_pos = np.arange(8)
        
        # モデル形式に応じた色
        color = 'red' if model_form == 'H' else 'blue'
        ax.bar(x_pos, E_vals_meV, width=0.6, label=f'{model_form}-form', color=color, alpha=0.7)
        
        ax.set_xlabel('Energy Level Index', fontsize=10)
        ax.set_ylabel('Energy [meV]', fontsize=10)
        ax.set_title(f"{data['label']} - Energy Levels (Absolute)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    for j in range(i+1, len(axes_e)):
        axes_e[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'energy_levels_{model_form}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ energy_levels_{model_form}.png saved")
    
    # 占有確率プロット（8準位別々のグラフ）
    fig_pop, axes_p = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes_p = axes_p.flatten() if n_datasets > 1 else [axes_p]
    
    for i, (data, ax) in enumerate(zip(datasets, axes_p)):
        # ハミルトニアン構築
        H_ham = uwbf.get_hamiltonian(data['B'], g, B4, B6)
        E_vals, _ = np.linalg.eigh(H_ham)
        
        # Boltzmann分布の計算には相対的エネルギーを使用
        E_vals_rel = E_vals - E_vals.min()
        boltzmann = np.exp(-E_vals_rel / data['T'])
        pops = boltzmann / boltzmann.sum()
        
        x_pos = np.arange(8)
        
        # モデル形式に応じた色
        color = 'red' if model_form == 'H' else 'blue'
        ax.bar(x_pos, pops, width=0.6, label=f'{model_form}-form', color=color, alpha=0.7)
        
        ax.set_xlabel('Energy Level Index', fontsize=10)
        ax.set_ylabel('Population', fontsize=10)
        ax.set_title(f"{data['label']} - Populations (T={data['T']:.1f}K)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # 基底状態占有率を表示
        ax.text(0.02, 0.98, f"Ground state: {pops[0]:.3f}", 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for j in range(i+1, len(axes_p)):
        axes_p[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'populations_{model_form}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ populations_{model_form}.png saved")

def plot_energy_levels_and_populations_comparison(datasets, params_H, params_B, output_dir):
    """エネルギー固有値と占有確率の比較プロット（H形式 vs B形式）"""
    # H形式パラメータ
    global_H, gamma_H = unpack_shared_gamma_parameters(params_H)
    g_H = global_H['g'] / SCALING_FACTORS['g']
    B4_H = global_H['B4'] / SCALING_FACTORS['B4']
    B6_H = global_H['B6'] / SCALING_FACTORS['B6']
    
    # B形式パラメータ
    global_B, gamma_B = unpack_shared_gamma_parameters(params_B)
    g_B = global_B['g'] / SCALING_FACTORS['g']
    B4_B = global_B['B4'] / SCALING_FACTORS['B4']
    B6_B = global_B['B6'] / SCALING_FACTORS['B6']
    
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))
    
    # エネルギー準位比較プロット
    fig_energy, axes_e = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes_e = axes_e.flatten() if n_datasets > 1 else [axes_e]
    
    for i, (data, ax) in enumerate(zip(datasets, axes_e)):
        # H形式
        H_ham_H = uwbf.get_hamiltonian(data['B'], g_H, B4_H, B6_H)
        E_vals_H, _ = np.linalg.eigh(H_ham_H)
        E_vals_H_meV = E_vals_H * 0.0862  # K → meV
        
        # B形式
        H_ham_B = uwbf.get_hamiltonian(data['B'], g_B, B4_B, B6_B)
        E_vals_B, _ = np.linalg.eigh(H_ham_B)
        E_vals_B_meV = E_vals_B * 0.0862  # K → meV
        
        x_pos = np.arange(8)
        width = 0.35
        
        ax.bar(x_pos - width/2, E_vals_H_meV, width, label='H-form', color='red', alpha=0.7)
        ax.bar(x_pos + width/2, E_vals_B_meV, width, label='B-form', color='blue', alpha=0.7)
        
        ax.set_xlabel('Energy Level Index', fontsize=10)
        ax.set_ylabel('Energy [meV]', fontsize=10)
        ax.set_title(f"{data['label']} - Energy Levels (Absolute)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    for j in range(i+1, len(axes_e)):
        axes_e[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_levels_HB_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ energy_levels_HB_comparison.png saved")
    
    # 占有確率比較プロット
    fig_pop, axes_p = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes_p = axes_p.flatten() if n_datasets > 1 else [axes_p]
    
    for i, (data, ax) in enumerate(zip(datasets, axes_p)):
        # H形式
        H_ham_H = uwbf.get_hamiltonian(data['B'], g_H, B4_H, B6_H)
        E_vals_H, _ = np.linalg.eigh(H_ham_H)
        E_vals_H_rel = E_vals_H - E_vals_H.min()
        boltzmann_H = np.exp(-E_vals_H_rel / data['T'])
        pops_H = boltzmann_H / boltzmann_H.sum()
        
        # B形式
        H_ham_B = uwbf.get_hamiltonian(data['B'], g_B, B4_B, B6_B)
        E_vals_B, _ = np.linalg.eigh(H_ham_B)
        E_vals_B_rel = E_vals_B - E_vals_B.min()
        boltzmann_B = np.exp(-E_vals_B_rel / data['T'])
        pops_B = boltzmann_B / boltzmann_B.sum()
        
        x_pos = np.arange(8)
        width = 0.35
        
        ax.bar(x_pos - width/2, pops_H, width, label='H-form', color='red', alpha=0.7)
        ax.bar(x_pos + width/2, pops_B, width, label='B-form', color='blue', alpha=0.7)
        
        ax.set_xlabel('Energy Level Index', fontsize=10)
        ax.set_ylabel('Population', fontsize=10)
        ax.set_title(f"{data['label']} - Populations (T={data['T']:.1f}K)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # 基底状態占有率を表示
        ax.text(0.02, 0.98, f"Ground (H): {pops_H[0]:.3f}\nGround (B): {pops_B[0]:.3f}", 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for j in range(i+1, len(axes_p)):
        axes_p[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'populations_HB_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ populations_HB_comparison.png saved")

# ==========================================
# 📊 Main Execution
# ==========================================
def main():
    print("="*80)
    print("Global Fitting v6: Shared Gamma Model (全データセット共通)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"パラメータ数：75個 → 12個（84%削減）")
    print("="*80)
    
    # データロード
    datasets = load_all_datasets(TARGET_DATA)
    
    if not datasets:
        print("❌ データロード失敗")
        return
    
    # ポラリトンモード検出と周波数ごとの重み配列生成
    print("\n🔍 Detecting polariton modes and assigning frequency-specific weights...")
    for data in datasets:
        # 各データセットでポラリトンと共振器領域を検出
        polariton_regions, cavity_regions = detect_peaks_and_classify(data['freq'], data['trans'])
        
        # 周波数ごとの重み配列を初期化（デフォルト: 1.0）
        weight_array = np.ones_like(data['freq'])
        
        # ポラリトン領域に1.5×の重みを適用
        for f_start, f_end in polariton_regions:
            mask = (data['freq'] >= f_start) & (data['freq'] <= f_end)
            weight_array[mask] = 1.5
        
        data['weight_array'] = weight_array
        data['polariton_regions'] = polariton_regions
        data['cavity_regions'] = cavity_regions
        
        # 統計情報の表示
        n_polariton = np.sum(weight_array > 1.0)
        n_total = len(weight_array)
        if n_polariton > 0:
            print(f"  ✓ {data['label']}: {n_polariton}/{n_total} points with polariton weight (1.5×)")
        else:
            print(f"  - {data['label']}: No polariton mode detected (all 1.0×)")
    
    # 初期値と境界値
    global_init, gamma_init = generate_shared_gamma_initial_values()
    params_init_scaled = pack_shared_gamma_parameters(global_init, gamma_init)
    lower_b, upper_b = get_shared_gamma_bounds()
    
    print(f"\n初期パラメータ数: {len(params_init_scaled)}")
    print(f"  Global: 5 (g, a, B4, B6, eps)")
    print(f"  Shared Gamma: 7 (全データセット共通)")
    print(f"  データセット数: {len(datasets)}")
    print(f"  データ点数: {sum(len(d['freq']) for d in datasets)}")
    print(f"  パラメータ/データ比: {len(params_init_scaled) / sum(len(d['freq']) for d in datasets) * 100:.2f}%")
    
    # 両モデル形式の最適化結果を保存
    results_by_form = {}
    
    for model_form in MODEL_FORMS:
        print(f"\n{'='*80}")
        print(f"🔄 Model: {model_form}-form")
        print(f"{'='*80}")
        
        # 3段階最適化
        print("\n🚀 Stage 1: Quick Exploration...")
        res_stage1 = least_squares(
            lambda p: shared_gamma_residuals(p, datasets, model_form),
            params_init_scaled,
            bounds=(lower_b, upper_b),
            max_nfev=5000,
            ftol=1e-5,
            xtol=1e-5,
            verbose=1
        )
        print(f"  Stage 1 Cost: {res_stage1.cost:.6e}")
        
        print("\n🚀 Stage 2: Medium Refinement...")
        res_stage2 = least_squares(
            lambda p: shared_gamma_residuals(p, datasets, model_form),
            res_stage1.x,
            bounds=(lower_b, upper_b),
            max_nfev=15000,
            ftol=1e-7,
            xtol=1e-7,
            verbose=1
        )
        print(f"  Stage 2 Cost: {res_stage2.cost:.6e}")
        
        print("\n🚀 Stage 3: Fine Tuning...")
        res_final = least_squares(
            lambda p: shared_gamma_residuals(p, datasets, model_form),
            res_stage2.x,
            bounds=(lower_b, upper_b),
            max_nfev=30000,
            ftol=1e-9,
            xtol=1e-9,
            verbose=2
        )
        print(f"  Final Cost: {res_final.cost:.6e}")
        print(f"  Total improvement: {(1 - res_final.cost/res_stage1.cost)*100:.1f}%")
        
        # 結果解析
        global_scaled, gamma_shared = unpack_shared_gamma_parameters(res_final.x)
        
        global_phys = {
            'g': global_scaled['g'] / SCALING_FACTORS['g'],
            'a': global_scaled['a'] / SCALING_FACTORS['a'],
            'B4': global_scaled['B4'] / SCALING_FACTORS['B4'],
            'B6': global_scaled['B6'] / SCALING_FACTORS['B6'],
            'eps': global_scaled['eps'] / SCALING_FACTORS['eps']
        }
        gamma_phys = gamma_shared / SCALING_FACTORS['gamma']
        
        print("\n" + "="*80)
        print("✅ Optimization Complete")
        print("="*80)
        print(f"Final Cost: {res_final.cost:.6e}")
        print(f"Iterations: {res_final.nfev}")
        print("-" * 80)
        print("【Global Parameters】")
        for key, val in global_phys.items():
            print(f"  {key:10s}: {val:12.8f}")
        
        print("\n【Shared Gamma (材料固有値)】")
        for i, g in enumerate(gamma_phys, 1):
            print(f"  γ{i}: {g:.6f} THz")
        
        # 条件数計算
        print("\n" + "="*80)
        print("📐 Condition Number Analysis")
        print("="*80)
        try:
            J = res_final.jac
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
            condition_number = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
            
            print(f"  Jacobian shape: {J.shape}")
            print(f"  Parameters: {J.shape[1]} (vs. 75 in v4)")
            print(f"  Max singular value: {s[0]:.4e}")
            print(f"  Min singular value: {s[-1]:.4e}")
            print(f"  Condition number: {condition_number:.4e}")
            
            if condition_number < 1e6:
                print("  ✅ Well-conditioned!")
            elif condition_number < 1e9:
                print("  ⚠️ Moderately ill-conditioned")
            else:
                print("  ❌ Ill-conditioned")
            
            # v4との比較
            print(f"\n  📊 Comparison:")
            print(f"    v4: κ ≈ 1.2×10¹⁶ (75 params, Cost=27,519)")
            print(f"    v5: κ = ∞ (21 params, Cost=29,112)")
            print(f"    v6: κ ≈ {condition_number:.2e} (12 params, Cost={res_final.cost:.0f})")
            
            if condition_number < 1.2e16:
                improvement = 1.2e16 / condition_number
                print(f"    Improvement: {improvement:.2e}×")
            
        except Exception as e:
            print(f"  ⚠️ Condition number calculation failed: {e}")
            condition_number = None
        
        # フィット品質
        fit_stats = analyze_shared_gamma_fit_quality(datasets, res_final.x, model_form)
        print("\n" + "-" * 80)
        print("【Fit Quality Statistics】")
        print(fit_stats.to_string(index=False))
        
        # 物理的診断
        diag_df = diagnose_problematic_regions(datasets, res_final.x, model_form)
        
        # 結果を保存
        results_by_form[model_form] = res_final.x
        
        # 出力
        out_dir = f"global_fitting_results_{model_form}_v6"
        os.makedirs(out_dir, exist_ok=True)
        
        fit_stats.to_csv(os.path.join(out_dir, 'fit_statistics.csv'), index=False)
        diag_df.to_csv(os.path.join(out_dir, 'diagnostic_analysis.csv'), index=False)
        
        with open(os.path.join(out_dir, 'shared_gamma_params.json'), 'w') as f:
            result_dict = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_form': model_form,
                'n_parameters': len(res_final.x),
                'final_cost': float(res_final.cost),
                'condition_number': float(condition_number) if condition_number is not None and np.isfinite(condition_number) else None,
                'global_params': {k: float(v) for k, v in global_phys.items()},
                'shared_gamma': gamma_phys.tolist()
            }
            json.dump(result_dict, f, indent=2)
        
        # プロット生成
        print("\n" + "-" * 80)
        print("📊 Generating plots...")
        plot_all_fits(datasets, res_final.x, out_dir, model_form)
        plot_residuals(datasets, res_final.x, out_dir, model_form)
        plot_chi_distribution(datasets, res_final.x, out_dir, model_form)
        
        # 新規追加: エネルギー準位と占有確率のプロット
        print("\n🔬 エネルギー準位・占有確率プロット生成中...")
        plot_energy_levels_and_populations(datasets, res_final.x, out_dir, model_form)
    
    # H形式とB形式の比較プロットを生成
    if 'H' in results_by_form and 'B' in results_by_form:
        print("\n" + "="*80)
        print("📊 Generating H-form vs B-form comparison plots...")
        print("="*80)
        comparison_dir = "global_fitting_results_comparison_v6"
        os.makedirs(comparison_dir, exist_ok=True)
        
        plot_all_fits_comparison(datasets, results_by_form['H'], results_by_form['B'], comparison_dir)
        
        # 新規追加: エネルギー準位と占有確率の比較プロット
        print("\n🔬 エネルギー準位・占有確率比較プロット生成中...")
        plot_energy_levels_and_populations_comparison(datasets, results_by_form['H'], results_by_form['B'], comparison_dir)
        
        print(f"\n✅ Comparison plots saved to: {comparison_dir}/")
        print("="*80)
        
        print(f"\n✅ Results saved to: {out_dir}/")
        print("="*80)

if __name__ == "__main__":
    main()
