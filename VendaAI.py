import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler




dados = pd.read_csv("https://gist.githubusercontent.com/guilhermesilveira/dd7ba8142321c2c8aaa0ddd6c8862fcc/raw/e694a9b43bae4d52b6c990a5654a193c3f870750/precos.csv")
## print(f"Original \n {dados.head()}")

dados["km_por_ano"] = dados["milhas_por_ano"] * 1.60934
## print(f"mi -> km \n {dados.head()}")

dados["tempo_de_uso"] = datetime.today().year - dados["ano_do_modelo"]
## print(f"tempo de uso \n {dados.head()}")


dados.drop(columns=["milhas_por_ano", "ano_do_modelo"],axis = 1, inplace=True)
print(f"Conteúdo tratado: \n {dados.head()} \n\n\n\n")


x = dados[["preco", "km_por_ano", "tempo_de_uso"]]
y = dados["vendido"]

SEED = 20

raw_treino_x, raw_teste_x, raw_treino_y, raw_teste_y = train_test_split(x, y, stratify=y, random_state=SEED)
print(f"Treinaremos com: {len(raw_treino_x)} e testaremos com: {len(raw_teste_y)} \n\n\n")

scaler = StandardScaler()
scaler.fit(raw_treino_x)

treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC(gamma = 'auto')
modelo.fit(treino_x, raw_treino_y)
previsoes = modelo.predict(teste_x)
acuracia = accuracy_score(raw_teste_y, previsoes)
print(f"A acurácia do modelo é: {acuracia:.2%}")

print(f"A taxa de carros vendidos é: {dados.query("vendido == True").shape[0] / len(dados):.2%}")

# DummyClassifier
from sklearn.dummy import DummyClassifier

SEED = 20

raw_treino_x, raw_teste_x, raw_treino_y, raw_teste_y = train_test_split(x, y, test_size=0.2, random_state=SEED)
print(f"Treinaremos com: {len(raw_treino_x)} e testaremos com: {len(raw_teste_y)} \n\n\n")

scaler = StandardScaler()
scaler.fit(raw_treino_x)

treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

classificador = DummyClassifier(strategy="most_frequent")
classificador.fit(treino_x, raw_treino_y)
previsoes = classificador.predict(teste_x)
acuracia = accuracy_score(raw_teste_y, previsoes)
print(f"A acurácia do DummyClassifier(strategy='most_frequent') é: {acuracia:.2%}")

# DummyClassifier -> strategy="stratified"
SEED = 20

raw_treino_x, raw_teste_x, raw_treino_y, raw_teste_y = train_test_split(x, y, test_size=0.2, random_state=SEED)
print(f"Treinaremos com: {len(raw_treino_x)} e testaremos com: {len(raw_teste_y)} \n\n\n")

scaler = StandardScaler()
scaler.fit(raw_treino_x)

treino_x = scaler.transform(raw_treino_x) 
teste_x = scaler.transform(raw_teste_x)

classificador = DummyClassifier(strategy="stratified")
classificador.fit(treino_x, raw_treino_y)
previsoes = classificador.predict(teste_x)
acuracia = accuracy_score(raw_teste_y, previsoes)
print(f"A acurácia do DummyClassifier(strategy='stratified') é : {acuracia:.2%}")