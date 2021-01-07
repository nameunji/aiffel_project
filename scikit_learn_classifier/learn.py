from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def learn_predict(model, data, label):
    if model == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=25)
        classifier = RandomForestClassifier(random_state=32)
    else:
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=7)

        if model == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            classifier = DecisionTreeClassifier(random_state=32)
        elif model == 'svm':
            from sklearn import svm
            classifier = svm.SVC()
        elif model == 'sgd':
            from sklearn.linear_model import SGDClassifier
            classifier = SGDClassifier()
        elif model == 'logistic':
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression(max_iter=3000)
        else:
            print("모델을 다시 선택해주세요.")

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f'modle : {model}')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return None