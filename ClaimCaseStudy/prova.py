
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X = [
    [110, 1.70, 'rugby'],
    [100, 1.90, 'basket'],
    [120, 1.90, 'rugby'],
    [ 70, 1.60, 'soccer']
]

trasformers = [
    ['category vectiruzer', OneHotEncoder(),[2]]
]
#remainder serve a tenere il resto delle colonne non specificate nella riga sopra
ct = ColumnTransformer(trasformers, remainder='passthrough')

#tranform oper trasformazioni sui dati
#in pratica da 1 0 0 alla prima, 0 1 0 alla seconda e 0 0 1 alla terza categoria
ct.fit(X)
X = ct.transform(X)
#ct.fit_transform(X) è la stessa cosa

print(X)