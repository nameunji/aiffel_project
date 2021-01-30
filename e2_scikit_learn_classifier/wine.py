import warnings
from sklearn.datasets import load_wine
from e2_scikit_learn_classifier.learn import learn_predict


warnings.filterwarnings("ignore")

# 데이터 준비
wines = load_wine()
wines_data = wines.data         # Feature Data 지정하기
wines_label = wines.target      # Label Data 지정하기


# 모델별 학습 및 예측
model_list = ['decision_tree', 'random_forest', 'svm', 'sgd', 'logistic']
for model in model_list:
    learn_predict(model, wines_data, wines_label)