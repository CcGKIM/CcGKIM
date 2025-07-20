# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import optuna

# 데이터 로드
df = pd.read_csv('./2023_국토교통부_매물별_전월세_실거래가_outrem.csv')
df.info()

# 권역 - 자치구 매핑
df['권역'] = df['자치구명'].map({
    "도심권": ["종로구", "중구", "용산구"],
    "동북권": ["성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구"],
    "서북권": ["은평구", "서대문구", "마포구"],
    "서남권": ["강서구", "양천구", "영등포구", "구로구", "금천구", "관악구", "동작구"],
    "동남권": ["서초구", "강남구", "송파구", "강동구"]
})

# 계약일자 처리
df['기준 월'] = pd.to_datetime(df['계약일자'], format='%Y%m%d').dt.strftime('%Y-%m')

# 결측치 처리
df["건축년도"].fillna(df.groupby("자치구명")["건축년도"].transform("median"), inplace=True)
df["층"].fillna(df.groupby("면적")["층"].transform("median"), inplace=True)
df["계약월수"].fillna(df.groupby("자치구명")["계약월수"].transform("mean"), inplace=True)

# 전세 -> 월세 변환
def wolJonConverter(x):
    if pd.isna(x['전월세전환율']) or pd.isna(x['월차임']):
        return np.nan
    return (x['월차임'] * 12) / (x['전월세전환율'] / 100) + x['보증금']

df['환산보증금'] = df.apply(wolJonConverter, axis=1)
df['로그환산보증금'] = np.log1p(df['환산보증금'])

# 이상치 제거 (K-Means 기반)
features = ['로그환산보증금', '면적', '계약월수']
X_scaled = StandardScaler().fit_transform(df[features])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df['distance_to_center'] = np.linalg.norm(X_scaled - kmeans.cluster_centers_[kmeans.labels_], axis=1)
threshold = np.percentile(df['distance_to_center'], 99.75)
df_cleaned = df[df['distance_to_center'] <= threshold]

# Feature 및 모델링 준비
X = pd.get_dummies(df_cleaned.drop(columns=['cluster', '로그환산보증금']), columns=['권역', '주택유형', '전월세구분'])
y = df_cleaned['로그환산보증금']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의 및 학습
xgb_model = XGBRegressor(n_estimators=1154, learning_rate=0.049, max_depth=8, random_state=42)
lgb_model = LGBMRegressor(n_estimators=800, learning_rate=0.031, max_depth=6, random_state=42)
stacked_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
    final_estimator=Ridge(alpha=6.26)
)
stacked_model.fit(X_train, y_train)

# 평가
y_pred = stacked_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R-squared: {r2:.4f}")

# Feature Importance 시각화
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

def plot_feature_importance(model, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(X.columns)[indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()

plot_feature_importance(xgb_model, "XGBoost Feature Importance")
plot_feature_importance(lgb_model, "LightGBM Feature Importance")
