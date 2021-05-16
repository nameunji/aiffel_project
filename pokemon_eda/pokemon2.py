import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


csv_path = os.getenv("HOME") +"/aiffel/pokemon_eda/data/Pokemon.csv"
original_data = pd.read_csv(csv_path)

# 컬럼 중 의미 없는 컬럼인 #와 문자열 데이터인 Name, Type 1, Type 2 데이터는 제외하고 사용
# target 데이터인 Legendary 또한 빼준다.
features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']

target = 'Legendary'
X = original_data[features]  # 훈련 데이터
y = original_data[target]    # 라벨


# 훈련 데이터와 학습 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
print(X_train.shape, y_train.shape)  # (640, 8) (640,)
print(X_test.shape, y_test.shape)    # (160, 8) (160,)

# 의사 결정 트리(decision tree) 사용해서 모델 정의 / random_state는 모델의 랜덤성을 제어
model = DecisionTreeClassifier(random_state=25)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 결과 확인
# 1. confusion_matrix
confusion_matrix(y_test, y_pred)
# TN, FP, FN, TP
# Positive는 Legendary=True(전설의 포켓몬), Negative는 Legendary=False(일반 포켓몬)
# TN (True Negative) : 옳게 판단한 Negative, 즉 일반 포켓몬을 일반 포켓몬이라고 알맞게 판단한 경우
# FP (False Positive) : 틀리게 판단한 Positive, 즉 일반 포켓몬을 전설의 포켓몬이라고 잘못 판단한 경우
# FN (False Negative) : 틀리게 판단한 Negative, 즉 전설의 포켓몬을 일반 포켓몬이라고 잘못 판단한 경우
# TP (True Positive) : 옳게 판단한 Positive, 즉 전설의 포켓몬을 전설의 포켓몬이라고 알맞게 판단한 경우


# 2. classification_report
print(classification_report(y_test, y_pred))