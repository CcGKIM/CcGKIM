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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 데이터 로드
path = "./data"
pd.set_option("display.max_columns", None)

rent = pd.read_csv(os.path.join(path, "2023_국토교통부_매물별_전월세_실거래가_outrem.csv"))

def load_conversion_rates():
    # 건물유형별 전월세전환율 데이터 로드
    dandok_conv = pd.read_csv(os.path.join(path, "dandok_conversion.csv"))
    apt_conv = pd.read_csv(os.path.join(path, "apt_conversion.csv"))
    yonda_conv = pd.read_csv(os.path.join(path, "yonda_conversion.csv"))
    off_conv = pd.read_csv(os.path.join(path, "off_conversion.csv"))
    off_conv.rename(columns={"그룹": "권역"}, inplace=True)
    return dandok_conv, apt_conv, yonda_conv, off_conv

dandok_conv, apt_conv, yonda_conv, off_conv = load_conversion_rates()

gu_to_area = {"종로구": "도심권", "중구": "도심권", "용산구": "도심권"}  # 예시 데이터
rent["권역"] = rent["자치구명"].map(gu_to_area)

def convRateMapper(row):
    tp = row["주택유형"]
    dt = int(row["계약일자"].replace("-", "")[:6])
    try:
        if tp == "아파트":
            return apt_conv.loc[(apt_conv["구"] == row["자치구명"]) & (apt_conv["연월"] == dt), "전월세전환율"].values[0]
        elif tp == "단독다가구":
            return dandok_conv.loc[(dandok_conv["권역"] == row["권역"]) & (dandok_conv["연월"] == dt), "전월세전환율"].values[0]
        elif tp == "연립다세대":
            return yonda_conv.loc[(yonda_conv["권역"] == row["권역"]) & (yonda_conv["연월"] == dt), "전월세전환율"].values[0]
        elif tp == "오피스텔":
            return off_conv.loc[(off_conv["권역"] == row["권역"]) & (off_conv["연월"] == dt), "전월세전환율"].values[0]
    except IndexError:
        return np.nan

rent["전월세전환율"] = rent.apply(convRateMapper, axis=1)

# 전세/월세 변환
rent_w = rent[rent["전월세구분"] == "월세"].copy()
rent_j = rent[rent["전월세구분"] == "전세"].copy()
rent_w["환산보증금"] = rent_w["월차임"] * 12 / (rent_w["전월세전환율"] / 100) + rent_w["보증금"]
rent_j["환산보증금"] = rent_j["보증금"]
rent_conv = pd.concat([rent_j, rent_w], axis=0).sort_index()

# 데이터 전처리 및 모델 학습
temp = rent_conv[["자치구명", "법정동명", "주택유형", "평수", "계약기간", "환산보증금"]].dropna()
le = LabelEncoder()
temp["자치구명"] = le.fit_transform(temp["자치구명"])
temp["법정동명"] = le.fit_transform(temp["법정동명"])
temp["주택유형"] = le.fit_transform(temp["주택유형"])
temp["계약기간"] = le.fit_transform(temp["계약기간"])

X = temp.drop("환산보증금", axis=1)
y = temp["환산보증금"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LGBMRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 모델 성능 평가
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})
print(feature_importance.sort_values(by="Importance", ascending=False))
