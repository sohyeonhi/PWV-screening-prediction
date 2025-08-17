# pre_mlp.py
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import torch  

__all__ = [
    "load_data", "preprocess_raw_data",
    "create_MetS", "apply_flags_and_masking", "flag_extreme_liver_risk",
    "astmmetric_winds", "add_derived_variables", "merged_variables", "drop_variables",
]

def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:

    # 순위형 변수 정리(숫자화)
    for col in ["PHY_ACT", "SMOKE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = create_MetS(df)
    df = create_flags(df)
    df = flag_extreme_liver_risk(df)

    # 비대칭 winsorize 한계값 설정
    limits = {
        "TG": (0, 0.01),     # 상한만 잘라냄
        "Wt": (0.01, 0.01),  # 양측
        "CR": (0.01, 0.01),  # 양측
        "AST": (0, 0.01),    # 상한만
        "ALT": (0, 0.01),    # 상한만
        "RGTP": (0.01, 0.01) # 양측 
    }
    df = astmmetric_winds(df, limits)

    df = add_derived_variables(df)
    df = merged_variables(df)
    df = drop_variables(df)

    return df

# --------------------------
# Feature builders
# --------------------------
def create_MetS(df: pd.DataFrame) -> pd.DataFrame:
    """
    대사증후군(MetS) 더미 생성(조건≥3).
    조건:
      1) TG ≥ 150
      2) HDL: 남<40, 여<50
      3) SBP ≥ 130 또는 DBP ≥ 85
      4) GLUCOSE ≥ 100
    """
    required = ["TG", "SEX", "HDL", "SBP", "DBP", "GLUCOSE"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    conditions = [
        (df["TG"] >= 150),
        ((df["SEX"] == 0) & (df["HDL"] < 40)) | ((df["SEX"] == 1) & (df["HDL"] < 50)),
        (df["SBP"] >= 130) | (df["DBP"] >= 85),
        (df["GLUCOSE"] >= 100),
    ]
    df["MetS_count"] = sum(cond.fillna(False).astype(int) for cond in conditions)
    df["MetS"] = (df["MetS_count"] >= 3).astype(int)
    df = df.drop(["MetS_count"], axis=1)
    return df

def create_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    임상 극단값 플래그 생성.
    """
    for c in ["TG", "HDL", "LDL", "T_CHOL", "GLUCOSE", "SBP", "DBP"]:
        if c not in df.columns:
            df[c] = np.nan

    df["extreme_TG_flag"] = (df["TG"] >= 500).fillna(False).astype(int)
    df["extreme_HDL_flag"] = (df["HDL"] < 40).fillna(False).astype(int)
    df["extreme_LDL_flag"] = (df["LDL"] >= 190).fillna(False).astype(int)
    df["extreme_TCHOL_flag"] = (df["T_CHOL"] >= 240).fillna(False).astype(int)

    df["extreme_GLUCOSE_flag"] = (df["GLUCOSE"] >= 126).fillna(False).astype(int)
    df["extreme_SBP_flag"] = (df["SBP"] >= 140).fillna(False).astype(int)
    df["extreme_DBP_flag"] = (df["DBP"] >= 90).fillna(False).astype(int)
    return df

def flag_extreme_liver_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    간 효소 위험 패턴 flag.
     - ALT>40 & AST/ALT<1 (NAFLD 패턴)
     - AST>40 & AST/ALT>2 (ALD 패턴)
     - ALT>40 & 성별 기준 GGT(=RGTP) 이상
     - AST>40 & AST/ALT>2 & GGT 이상
    """
    for c in ["AST", "ALT", "RGTP", "SEX"]:
        if c not in df.columns:
            df[c] = np.nan

    ratio = df["AST"] / df["ALT"].replace(0, np.nan)
    ggt_abnormal = (
        ((df["SEX"] == 0) & (df["RGTP"] > 73)) |
        ((df["SEX"] == 1) & (df["RGTP"] > 48))
    )

    condition1 = (df["ALT"] > 40) & (ratio > -np.inf) & (ratio < 1.0)
    condition2 = (df["AST"] > 40) & (ratio > 2.0)
    condition3 = (df["ALT"] > 40) & ggt_abnormal
    condition4 = (df["AST"] > 40) & (ratio > 2.0) & ggt_abnormal

    df["extreme_LiverRisk_flag"] = (condition1 | condition2 | condition3 | condition4).fillna(False).astype(int)
    return df

def astmmetric_winds(df: pd.DataFrame, limits_dict: dict) -> pd.DataFrame:
    """
    각 컬럼별 winsorize 적용. limits_dict 예: {'TG': (0,0.01)}
    """
    for col, limits in limits_dict.items():
        if col in df.columns:
            series = df[col].astype(float)
            clipped = winsorize(series, limits=limits)
            df[col] = pd.Series(clipped.data, index=df.index)
        else:
            print(f"[경고] '{col}' 컬럼이 데이터프레임에 없습니다.")
    return df

def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    - BMI, LDL/HDL, AIP, Non-HDL, Remnant, BMI_outlier_flag
    - TG/HDL은 중간 계산 후 삭제(요청 원본 유지)
    """
    for c in ["Wt", "Ht", "TG", "HDL", "LDL", "T_CHOL"]:
        if c not in df.columns:
            df[c] = np.nan

    # BMI
    height_m = (df["Ht"] * 0.01)
    df["BMI"] = df["Wt"] / (height_m.replace(0, np.nan) ** 2)
    #  Ht, Wt 제거
    df = df.drop(["Ht", "Wt"], axis=1, errors="ignore")

    # TG/HDL (0 분모 방지)
    df["TG/HDL"] = df["TG"] / df["HDL"].replace(0, np.nan)

    # LDL/HDL
    df["LDL/HDL"] = df["LDL"] / df["HDL"].replace(0, np.nan)

    # AIP = log10(TG/HDL)
    df["AIP"] = np.log10(df["TG/HDL"])

    # Non-HDL = TCHOL - HDL
    df["Non_HDL"] = df["T_CHOL"] - df["HDL"]

    # BMI 이상치 flag
    df["BMI_outlier_flag"] = (df["BMI"] >= 25).fillna(0).astype(int)

    # 중간변수 삭제
    df = df.drop(["TG/HDL"], axis=1, errors="ignore")
    return df

def merged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """과거력 합산 점수."""
    past_history = ["HX_STROKE", "HX_MI", "HX_HTN", "HX_DYSLI", "HX_ATHERO", "HX_DM"]
    for c in past_history:
        if c not in df.columns:
            df[c] = 0
    df["PAST_HISTORY_SCORE"] = df[past_history].sum(axis=1)
    return df

def drop_variables(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "IDX", "mean_PWV",
        "HX_STROKE", "HX_MI", "HX_HTN", "HX_DM", "HX_DYSLI", "HX_ATHERO"
    ] 
    # "T_CHOL", "HDL", "LDL", "TG"
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df

if __name__ == "__main__":
    import sys, os, inspect
    print("[DEBUG] pre_mlp file:", __file__)
    print("[DEBUG] has load_data:", "load_data" in globals())
    print("[DEBUG] exported:", __all__)

    pass
