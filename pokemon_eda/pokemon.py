import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


csv_path = os.getenv("HOME") +"/aiffel/pokemon_eda/data/Pokemon.csv"
original_data = pd.read_csv(csv_path)
pokemon = original_data.copy()

legendary = pokemon[pokemon["Legendary"] == True].reset_index(drop=True)  # 전설의 포켓몬 데이터셋
ordinary = pokemon[pokemon["Legendary"] == False].reset_index(drop=True)  # 일반 포켓몬 데이터셋
stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
types = list(set(pokemon["Type 1"]))

"""
step 1. 데이터 전처리
1. 이름의 길이 : name_count 컬럼을 생성 후 길이가 10을 넘는지 아닌지에 대한 categorical 컬럼을 생성
2. 토큰 추출 : legendary 포켓몬에서 많이 등장하는 토큰을 추려내고 토큰 포함 여부를 원-핫 인코딩(One-Hot Encoding)으로 처리
    포켓몬의 이름은 총 네 가지 타입으로 나뉜다.
    (1) 한 단어면 ex. Venusaur
    (2) 두 단어이고, 앞 단어는 두 개의 대문자를 가지며 대문자를 기준으로 두 부분으로 나뉘는 경우 ex. VenusaurMega Venusaur
    (3) 이름은 두 단어이고, 맨 뒤에 X, Y로 성별을 표시하는 경우 ex. CharizardMega Charizard X
    (4) 알파벳이 아닌 문자를 포함하는 경우 ex. Zygarde50% Forme
"""
# (1). 각 이름 길이 name_count컬럼에 넣어주기
pokemon["name_count"] = pokemon["Name"].apply(lambda i: len(i))
legendary["name_count"] = legendary["Name"].apply(lambda i: len(i))
ordinary["name_count"] = ordinary["Name"].apply(lambda i: len(i))

# 이름의 길이가 10 이상이면 True, 미만이면 False를 가지는 long_name 컬럼을 생성
pokemon["long_name"] = pokemon["name_count"] >= 10

# (2) 이름에 자주 쓰이는 토큰 추출
# 이름에 알파벳이 아닌 문자가 들어간 경우 전처리하기
# 띄어쓰기가 있는 경우에도 isalpha() = False로 처리되기때문에, 알파벳 체크를 위해 띄어쓰기가 없는 컬럼을 따로 만들어준 후, 띄어쓰기를 빈칸으로 처리해서 확인
pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())

# alpha가 아닌 데이터 확인해보기 -> 9마리 뿐이라 수동으로 다 바꿔주기
print(pokemon[pokemon["name_isalpha"] == False].shape)  # (9, 17)
print(pokemon[pokemon["name_isalpha"] == False])
pokemon = pokemon.replace(to_replace="Nidoran♀", value="Nidoran X")
pokemon = pokemon.replace(to_replace="Nidoran♂", value="Nidoran Y")
pokemon = pokemon.replace(to_replace="Farfetch'd", value="Farfetchd")
pokemon = pokemon.replace(to_replace="Mr. Mime", value="Mr Mime")
pokemon = pokemon.replace(to_replace="Porygon2", value="Porygon")
pokemon = pokemon.replace(to_replace="Ho-oh", value="Ho Oh")
pokemon = pokemon.replace(to_replace="Mime Jr.", value="Mime Jr")
pokemon = pokemon.replace(to_replace="Porygon-Z", value="Porygon Z")
pokemon = pokemon.replace(to_replace="Zygarde50% Forme", value="Zygarde Forme")

pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())

# 이름을 띄어쓰기 & 대문자 기준으로 분리해 토큰화하기
def tokenize(name):
    name_split = name.split(" ")

    tokens = []
    for part_name in name_split:
        a = re.findall('[A-Z][a-z]*', part_name)
        tokens.extend(a)

    return np.array(tokens)

all_tokens = list(legendary["Name"].apply(tokenize).values)

token_set = []
for token in all_tokens:
    token_set.extend(token)

print(len(set(token_set)))  # 65
print(token_set)

# 많이 사용된 토큰 추출
most_common = Counter(token_set).most_common(10)
# 전설의 포켓몬 이름에 등장하는 토큰이 포켓몬의 이름에 있는지의 여부를 나타내는 컬럼
for token, _ in most_common:
    pokemon[token] = pokemon["Name"].str.contains(token)


"""
step 2. Type1 & 2! 범주형 데이터 전처리하기
"""
# 18개의 컬럼에 대해 원-핫 인코딩 작업
for t in types:
    pokemon[t] = (pokemon["Type 1"] == t) | (pokemon["Type 2"] == t)
pokemon[[["Type 1", "Type 2"] + types][0]].head()  # 출력하여 체크



features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation',
            'name_count', 'long_name', 'Forme', 'Mega', 'Mewtwo', 'Kyurem', 'Deoxys', 'Hoopa',
            'Latias', 'Latios', 'Kyogre', 'Groudon', 'Poison', 'Water', 'Steel', 'Grass',
            'Bug', 'Normal', 'Fire', 'Fighting', 'Electric', 'Psychic', 'Ghost', 'Ice',
            'Rock', 'Dark', 'Flying', 'Ground', 'Dragon', 'Fairy']
target = "Legendary"
X = pokemon[features]
y = pokemon[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
model = DecisionTreeClassifier(random_state=25)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))