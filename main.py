import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

print("=== CARREGAMENTO E PREPARAÇÃO DOS DADOS ===")

# Carregamento dos dados
dados = pd.read_csv("https://gist.githubusercontent.com/guilhermesilveira/dd7ba8142321c2c8aaa0ddd6c8862fcc/raw/e694a9b43bae4d52b6c990a5654a193c3f870750/precos.csv")
print(f"Dados originais:\n{dados.head()}\n")

# Conversão de milhas para quilômetros
dados["km_por_ano"] = dados["milhas_por_ano"] * 1.60934
print(f"Após conversão milhas -> km:\n{dados.head()}\n")

# Cálculo da idade do veículo
dados["idade"] = datetime.today().year - dados["ano_do_modelo"]
print(f"Após cálculo da idade:\n{dados.head()}\n")

# Limpeza dos dados
dados.drop(["milhas_por_ano", "ano_do_modelo"], axis=1, inplace=True)
print(f"Dados finais após limpeza:\n{dados.head()}\n")

# Definição das variáveis X e Y
x = dados[["preco", "idade", "km_por_ano"]]
y = dados["vendido"]

# Análise da proporção de carros vendidos
proporcao_vendidos = len(dados.query("vendido == True")) / len(dados)
print(f"Proporção de carros vendidos: {proporcao_vendidos:.2%}\n")

SEED = 20

print("=== TESTES COM DIFERENTES ALGORITMOS ===\n")

# ========================================
# 1. DUMMYCLASSIFIER - ESTRATÉGIA PADRÃO
# ========================================
print("1. DummyClassifier (estratégia padrão)")

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y,random_state=SEED,stratify=y)
print(f"Treinaremos com {len(raw_treino_x)}")
print(f"Testaremos com {len(raw_teste_x)}")

classificador = DummyClassifier()
classificador.fit(raw_treino_x, treino_y)
previsoes = classificador.predict(raw_teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"A acurácia do dummy foi de {acuracia:.2f}%\n")

# ========================================
# 2. DUMMYCLASSIFIER - ESTRATÉGIA ESTRATIFICADA
# ========================================
print("2. DummyClassifier (estratégia estratificada)")

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y,random_state=SEED,stratify=y)
print(f"Treinaremos com {len(raw_treino_x)}")
print(f"Testaremos com {len(raw_teste_x)}")

classificador = DummyClassifier(strategy='stratified')
classificador.fit(raw_treino_x, treino_y)
previsoes = classificador.predict(raw_teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"A acurácia do dummy foi de {acuracia:.2f}%\n")

# ========================================
# 3. SVM COM NORMALIZAÇÃO
# ========================================
print("3. SVM com normalização")

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y,random_state=SEED,stratify=y)
print(f"Treinaremos com {len(raw_treino_x)}")
print(f"Testaremos com {len(raw_teste_x)}")

scaler = StandardScaler()
scaler.fit(raw_treino_x)

treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC(gamma='auto')
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"A acurácia foi de {acuracia:.2f}%\n")

# ========================================
# 4. LINEAR SVC
# ========================================
print("4. LinearSVC")

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y,random_state=SEED,stratify=y)
print(f"Treinaremos com {len(raw_treino_x)}")
print(f"Testaremos com {len(raw_teste_x)}")

scaler = StandardScaler()
scaler.fit(raw_treino_x)

treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"A acurácia foi de {acuracia:.2f}%\n")

# ========================================
# 5. DECISION TREE (ÁRVORE DE DECISÃO)
# ========================================
print("5. Decision Tree (Árvore de Decisão)")

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,random_state=SEED,stratify=y)
print(f"Treinaremos com {len(treino_x)}")
print(f"Testaremos com {len(teste_x)}")

# Nota: Decision Tree não precisa de normalização
modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"A acurácia foi de {acuracia:.2f}%\n")

# ========================================
# 6. VISUALIZAÇÃO DA ÁRVORE DE DECISÃO
# ========================================
print("6. Gerando visualização da Árvore de Decisão")

try:
    estrutura = export_graphviz(modelo, filled=True, rounded=True,
                                feature_names=x.columns,
                                class_names=["não", "sim"])
    grafico = graphviz.Source(estrutura)
    
    # Salva a visualização em arquivo
    grafico.render('arvore_decisao', format='png', cleanup=True)
    print("Visualização da árvore salva como 'arvore_decisao.png'")
    print("Estrutura da árvore:")
    print(grafico.source)
    
except Exception as e:
    print(f"Erro ao gerar visualização: {e}")
    print("Certifique-se de que o Graphviz está instalado no sistema")

print("\n=== ANÁLISE COMPLETA FINALIZADA ===")
