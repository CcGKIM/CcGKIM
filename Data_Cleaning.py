# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc

# 한글 설정
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc("font", family=font_name)
elif platform.system() == "Darwin":
    rc("font", family="Arial Unicode MS")

# 데이터 로드
def load_data(filepath):
    df = pd.read_excel(filepath)
    df["계약일자"] = pd.to_datetime(df["계약일자"].astype(str), format="%Y%m%d")
    df["계약일자_연월"] = df["계약일자"].dt.to_period("M")
    return df

# 이상치 시각화 함수
def plot_outliers(df, column, threshold, title, color="blue"):
    temp_df = df[df[column] >= threshold]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("카운트")
    plt.hist(temp_df[column], bins=10, color=color)
    plt.show()

# 계약일자별 카운트 시각화
def plot_contract_counts(df):
    date_counts = df["계약일자_연월"].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    plt.plot(date_counts.index.astype(str), date_counts.values, marker="o", linestyle="-", color="orange", label="계약건수")
    plt.xlabel("계약일자")
    plt.ylabel("카운트")
    plt.title("계약일자별 거래량")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

# 특정 조건의 데이터 개수 반환
def count_filtered_data(df, deposit_threshold=6000, rent_threshold=30):
    return len(df[(df["보증금"] <= deposit_threshold) & (df["월차임"] <= rent_threshold)])

# 이상치 제거 함수 (IQR 방식)
def remove_outliers(df, column, weight=1.5, keep_null=False):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - weight * iqr
    upper_bound = q3 + weight * iqr
    
    if keep_null:
        return df[(df[column].isna()) | ((df[column] >= lower_bound) & (df[column] <= upper_bound))]
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 실행 예시
if __name__ == "__main__":
    filepath = "./2023_국토교통부_매물별_전월세_실거래가.xlsx"
    df = load_data(filepath)
    
    plot_outliers(df, "면적", 330, "330㎡ 이상 면적 시각화", color="green")
    plot_contract_counts(df)
    
    filtered_count = count_filtered_data(df)
    print(f"보증금 6000만원 이하, 월차임 30만원 이하 신고 대상 제외 데이터 개수: {filtered_count}")
    
    cleaned_df = remove_outliers(df, "보증금", weight=1.5, keep_null=True)
    print(f"이상치 제거 후 데이터 크기: {cleaned_df.shape}")
