# 필요한 라이브러리 임포트
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import json

# 데이터 경로 설정
path = "/content/drive/MyDrive/colab_notebooks/03_recollected_data"
pd.set_option("display.max_columns", None)

# 데이터 로드
rent = pd.read_csv(os.path.join(path, "2023_국토교통부_매물별_전월세_실거래가.csv"))

# 데이터 정보 확인
rent.info()
print(rent.isna().sum().sort_values(ascending=False))

# 보증금/월차임 결측치 점검
print(rent[(rent["보증금"]==0) & (rent["월차임"]==0)].shape)
print(rent[(rent["보증금"]==0)]["전월세구분"].value_counts(dropna=False))
print(rent[(rent["월차임"]==0)]["전월세구분"].value_counts(dropna=False))

# 본번/부번/층 결측치 점검
missing_bonbun = rent[(rent["본번"].isna()) & (rent["부번"].isna()) & (rent["층"].isna())]
print(missing_bonbun.shape)

# 건축년도 결측치 점검
print(rent[rent["건축년도"].isna()]["주택유형"].value_counts())

# 계약 시작/종료월 결측치 점검
print(rent[rent["계약시작월"].isna() | rent["계약종료월"].isna()]["주택유형"].value_counts())

# 이상치 점검
print(rent[rent["건축년도"] >= 2024])
print(rent[rent["층"] == 0])
print(rent[rent["층"] <= -1]["주택유형"].value_counts())

# 평수 계산 및 이상치 제거
def sizeConverter(val):
    return round(val / 3.30579)

rent["평수"] = rent["면적"].apply(sizeConverter)
rent = rent[rent["평수"] != 1]

# 단지명 정제
import re
rent["단지명"] = rent["단지명"].apply(lambda x: re.sub("[0-9]+-[0-9]+", "", str(x)))
rent["단지명"] = rent["단지명"].replace("", np.nan)

# 평수구간 파생변수 생성
def sizeChecker(val):
    if val <= 5:
        return "5평 이하"
    elif val <= 10:
        return "5~10평"
    elif val <= 20:
        return "10~20평"
    elif val <= 30:
        return "20~30평"
    else:
        return "30평 이상"

rent["평수구간"] = rent["평수"].apply(sizeChecker)

# 계약기간 파생변수 생성
rent["계약시작월"] = pd.to_datetime(rent["계약시작월"], format="%Y-%m")
rent["계약종료월"] = pd.to_datetime(rent["계약종료월"], format="%Y-%m")
rent["계약월수"] = ((rent["계약종료월"].dt.year - rent["계약시작월"].dt.year) * 12 + 
                    (rent["계약종료월"].dt.month - rent["계약시작월"].dt.month))

# 데이터 시각화
NUM_COLS = ["층", "평수", "건축년도", "보증금", "월차임"]
plt.figure(figsize=(7,6))
sns.heatmap(rent[NUM_COLS].corr(), annot=True, cmap="RdBu_r", fmt=".2f", vmin=-1, vmax=1)
plt.title("수치형 변수별 상관관계")
plt.show()

# 지도 시각화
geo_path = "/content/drive/MyDrive/colab_notebooks/02_cleaned_data/seoul_municipalities_geo_simple.json"
with open(geo_path) as f:
    geo_info = json.load(f)

plot_data = (rent.groupby("자치구명")["보증금"].median()/10_000).reset_index()
m = folium.Map(location=[37.55408, 126.9902], zoom_start=11, tiles="cartodbpositron")
c = folium.Choropleth(
    geo_data=geo_info,
    data=plot_data,
    columns=["자치구명", "보증금"],
    fill_color='YlOrRd',
    fill_opacity=0.5,
    line_opacity=0.2,
    key_on="properties.name",
    legend_name="전세 보증금 중위수 (억)"").add_to(m)
m.save("seoul_rent_map.html")
