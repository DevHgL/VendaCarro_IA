# VendaCarro_IA

## Descrição do Projeto

Este projeto utiliza Machine Learning para prever se um carro será vendido ou não, baseado em características como preço, quilometragem por ano e tempo de uso. O projeto implementa diferentes algoritmos de classificação para comparar suas performances.

## Estrutura do Projeto

### Arquivos

- **`main.py`**: Script principal unificado com todos os algoritmos de ML (RECOMENDADO)
- **`VendaAI.py`**: Script com análise usando SVM e comparação com DummyClassifier
- **`train_test.py`**: Implementação usando Decision Tree com visualização gráfica
- **`Introdução_a_Classificação_2024_video_5_3.ipynb`**: Notebook Jupyter original com análise interativa
- **`requirements.txt`**: Lista das dependências necessárias para executar o projeto

## Descrição dos Códigos

### main.py (RECOMENDADO)

Este é o arquivo principal consolidado que combina todos os algoritmos de Machine Learning em uma única execução:

**Algoritmos implementados:**

1. **DummyClassifier** (estratégia padrão e estratificada) - Modelos baseline
2. **SVM (Support Vector Machine)** - Classificação com kernel RBF
3. **LinearSVC** - Support Vector Classifier linear  
4. **DecisionTreeClassifier** - Árvore de decisão com visualização

**Funcionalidades:**

- Preparação completa dos dados (conversão, limpeza, transformação)
- Execução sequencial de todos os algoritmos
- Comparação de performance entre modelos
- Geração automática de visualização da árvore de decisão
- Relatório completo com acurácias de todos os modelos
- Prints informativos para acompanhamento do progresso

### VendaAI.py

Este arquivo implementa um modelo de classificação para prever vendas de carros usando:

**Preparação dos dados:**
- Carrega dados de um CSV online com informações de carros
- Converte milhas por ano para quilômetros por ano
- Calcula o tempo de uso baseado no ano atual
- Remove colunas desnecessárias

**Modelos implementados:**
1. **SVM (Support Vector Machine)**: Modelo principal de classificação
2. **DummyClassifier**: Modelo baseline para comparação
   - Strategy "most_frequent": Sempre prediz a classe mais comum
   - Strategy "stratified": Prediz respeitando a proporção das classes

**Funcionalidades:**
- Divisão treino/teste com estratificação
- Normalização dos dados usando StandardScaler
- Cálculo de acurácia dos modelos
- Comparação com taxa base de carros vendidos

### train_test.py

Este arquivo implementa um modelo usando Decision Tree (Árvore de Decisão):

**Características:**
- Usa os mesmos dados e preparação do VendaAI.py
- Implementa DecisionTreeClassifier com profundidade máxima de 3
- Gera visualização gráfica da árvore de decisão usando Graphviz
- Calcula e exibe a acurácia do modelo

**Visualização:**
- Exporta a estrutura da árvore de decisão
- Cria um gráfico visual mostrando as regras de decisão
- Nomeia as features e classes para melhor interpretabilidade

## Instalação e Configuração

### Pré-requisitos

- Python 3.7 ou superior
- pip (gerenciador de pacotes Python)

### Instalação das Dependências

1. Clone ou baixe o repositório
2. Navegue até o diretório do projeto
3. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```
4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Para usar graphviz (visualização da árvore)

No Ubuntu/Debian:
```bash
sudo apt-get install graphviz
```

No macOS:
```bash
brew install graphviz
```

No Windows:
- Baixe e instale o Graphviz do site oficial
- Adicione o diretório bin ao PATH do sistema

## Como Executar

### Executar análise completa (RECOMENDADO)

```bash
python main.py
```

Este comando executa todos os algoritmos sequencialmente e gera um relatório completo com:
- Comparação de acurácia entre todos os modelos
- Análise da qualidade dos dados
- Visualização da árvore de decisão (salva como PNG)

### Executar modelos individuais

**Modelo SVM:**
```bash
python VendaAI.py
```

**Modelo Decision Tree:**
```bash
python train_test.py
```

**Notebook interativo:**
```bash
jupyter notebook
# Abra o arquivo: Introdução_a_Classificação_2024_video_5_3.ipynb
```

## Resultados Esperados

- **main.py**: Análise completa com comparação de 5 algoritmos diferentes, incluindo acurácias e visualização da árvore de decisão
- **VendaAI.py**: Mostra a acurácia do SVM e compara com modelos baseline (DummyClassifier)
- **train_test.py**: Mostra a acurácia da Decision Tree e exibe a visualização da árvore
- **Notebook**: Análise interativa passo-a-passo com visualizações dos dados

## Dataset

O projeto utiliza um dataset público de vendas de carros disponível online, contendo:
- Preço do carro
- Milhas por ano (convertidas para km/ano)
- Ano do modelo (convertido para tempo de uso)
- Status de venda (vendido/não vendido)

## Tecnologias Utilizadas

- **Python**: Linguagem principal
- **Pandas**: Manipulação e análise de dados
- **Scikit-learn**: Algoritmos de Machine Learning
- **Graphviz**: Visualização de árvores de decisão

## Contribuições

Sinta-se à vontade para contribuir com melhorias, correções de bugs ou novas funcionalidades!