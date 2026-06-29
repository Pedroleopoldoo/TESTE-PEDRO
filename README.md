# Sistema de Recomendação com Deep Learning (PyTorch)

## Visão Geral

Este projeto implementa um **Sistema de Recomendação baseado em Filtragem Colaborativa utilizando Deep Learning**.

O modelo aprende representações vetoriais (embeddings) de usuários e produtos a partir do histórico de compras do dataset **Instacart** e estima a probabilidade de um determinado usuário comprar determinado produto.

---

# Tecnologias

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit (interface)

---

# Estrutura do Projeto

```
recommendation-system/

data/
    instacart_orders.parquet

models/
    best_model.pth
    user_encoder.pkl
    item_encoder.pkl

train.py
predict.py
app.py
README.md
requirements.txt
```

---

# Funcionamento

O pipeline do treinamento segue as etapas abaixo:

```
Leitura dos dados
        │
        ▼
Remoção de duplicidades
        │
        ▼
Amostragem de usuários
        │
        ▼
Codificação dos IDs
(LabelEncoder)
        │
        ▼
Geração de amostras negativas
        │
        ▼
Treino/Teste
        │
        ▼
Embeddings
        │
        ▼
Rede Neural
        │
        ▼
Treinamento
        │
        ▼
Avaliação
        │
        ▼
Salvamento do modelo
```

---

# Arquitetura da Rede

```
Usuário
        \
         \
          Embedding
         /
Produto

        │
        ▼

Concatenação

        │

Linear (128)

        │

ReLU

        │

Dropout

        │

Linear (64)

        │

ReLU

        │

Linear (1)

        │

Probabilidade
```

---

# Negative Sampling

O dataset original contém apenas interações positivas (compras).

Para que o modelo aprenda também o que **não recomendar**, são geradas amostras negativas.

Exemplo:

Positivos

| Usuário | Produto |
| ------- | ------- |
| João    | Leite   |
| João    | Arroz   |

Negativos

| Usuário | Produto |
| ------- | ------- |
| João    | Shampoo |
| João    | Café    |

---

# Treinamento

Durante o treinamento são utilizados:

* BCEWithLogitsLoss
* Adam Optimizer
* Batch Size = 1024
* Embeddings de dimensão 64

O melhor modelo é salvo automaticamente.

---

# Avaliação

São calculadas as métricas:

* AUC
* Accuracy
* Precision
* Recall
* F1 Score

---

# Arquivos Gerados

Após o treinamento são criados:

```
models/

best_model.pth
user_encoder.pkl
item_encoder.pkl
```

Esses arquivos são utilizados posteriormente durante as recomendações.

---

# Executando

Treinar o modelo

```bash
python train.py
```

Executar a interface

```bash
streamlit run app.py
```

---

# Objetivo

O objetivo deste projeto é demonstrar a construção completa de um Sistema de Recomendação utilizando Redes Neurais em PyTorch, desde o pré-processamento até a disponibilização do modelo em uma interface simples.
