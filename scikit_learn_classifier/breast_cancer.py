from sklearn.datasets import load_breast_cancer
from scikit_learn_classifier.learn import learn_predict


# 데이터 준비
cancer = load_breast_cancer()
cancer_data = cancer.data         # Feature Data 지정하기
cancer_label = cancer.target      # Label Data 지정하기


# 모델별 학습 및 예측
# model_list = ['decision_tree', 'random_forest', 'svm', 'sgd', 'logistic']
model_list = ['logistic']

for model in model_list:
    learn_predict(model, cancer_data, cancer_label)
