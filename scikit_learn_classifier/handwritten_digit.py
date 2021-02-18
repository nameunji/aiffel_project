from sklearn.datasets import load_digits
from scikit_learn_classifier.learn import learn_predict


# 데이터 준비
digits = load_digits()
digits_data = digits.data      # Feature Data 지정하기
digits_label = digits.target   # Label Data 지정하기


# 모델별 학습 및 예측
model_list = ['decision_tree', 'random_forest', 'svm', 'sgd', 'logistic']
for model in model_list:
    learn_predict(model, digits_data, digits_label)
