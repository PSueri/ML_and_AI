import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Caricamento del dataset
data = pd.read_csv('claims_data.csv')
print(data.info())

#################################
# Pulizia variabili categoriche #
#################################
# Claim Severity
print(data.claim_severity.value_counts())
severity_map_dict = {'Minor': 0,
                     'Major': 1,
                     'Catastrophic': 2
                     }
data['claim_severity'] = data['claim_severity'].map(severity_map_dict)

# Customer Gender
print(data.customer_gender.value_counts())
gender_map_dict = {'M': 0, 'F': 1}
data['customer_gender'] = data['customer_gender'].map(gender_map_dict)

# Customer Occupation
print(data.customer_occupation.value_counts())
occupation_map_dict = {'Unemployed': 0, 'Blue-collar': 1, 'Professional': 2}
data['customer_occupation'] = data['customer_occupation'].map(occupation_map_dict)

# Policy Type
print(data.policy_type.value_counts())
policy_map_dict = {'Standard': 0, 'Premium': 1}
data['policy_type'] = data['policy_type'].map(policy_map_dict)

# Claim Type
print(data.claim_type.value_counts())
data['claim_type_A'] = np.where(data['claim_type'] == 'Auto', 1, 0)
data['claim_type_L'] = np.where(data['claim_type'] == 'Life', 1, 0)
data['claim_type_H'] = np.where(data['claim_type'] == 'Home', 1, 0)

print(data.info())

#####################
# Divisione dataset #
#####################
# Divido features e labels
X = data[
    ['claim_type_A', 'claim_type_H', 'claim_type_L', 'claim_severity', 'customer_age', 'customer_gender',
     'customer_occupation', 'customer_claims_history', 'policy_type', 'policy_coverage', 'policy_deductible']]
y = data['claim_amount']

# Divisione del dataset in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#####################################
# Allenamento e valutazione modelli #
#####################################
# Inizializzazione dei modelli, aggiungere eventuale GridSearcgCV per hyper-parameter tuning
models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    KNeighborsRegressor(n_neighbors=5),
    RandomForestRegressor(n_estimators=100),
    MLPRegressor(hidden_layer_sizes=(100, 50))
]

# Addestramento e valutazione dei modelli
for model in models:
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"{model.__class__.__name__}: MAE = {mae:.2f}, MAPE = {mape:.2f}%, RMSE = {rmse:.2f}, R^2 = {r2:.2f}")

############################################
# Selezione e salvataggio modello migliore #
############################################
# Selezione del modello migliore in termini di mape
best_model = min(models, key=lambda m: mean_absolute_percentage_error(y_test, m.predict(X_test)))
print(f"Modello migliore: {best_model.__class__.__name__}")

# Salvataggio del modello migliore
pickle.dump(best_model, open('best_model.pkl', 'wb'))
