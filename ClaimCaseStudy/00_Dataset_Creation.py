import pandas as pd
import numpy as np
import random

def generate_claims_dataset(n_entries):
    """
    Questa funzione genera un dataset di esempio per la previsione di future richieste di risarcimento.

    Parametri:
        n_entries (int): Il numero di righe (entry) da generare nel dataset.

    Restituisce:
        pandas.DataFrame: Il dataset generato.
    """

    def generate_random_date(start_year, end_year):
        """Generates a random date between the specified years.

        Args:
          start_year: The starting year.
          end_year: The ending year.

        Returns:
          A pandas Timestamp object representing the random date.
        """

        year = np.random.randint(start_year, end_year + 1)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)  # Assuming 28 days for simplicity
        return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
    # Definizione delle colonne del dataset
    columns = [
        'claim_id', 'claim_date', 'claim_type', 'claim_severity', 'claim_amount',
        'customer_age', 'customer_gender', 'customer_occupation', 'customer_claims_history',
        'policy_type', 'policy_coverage', 'policy_deductible'
    ]

    # Generazione di dati random per ogni colonna
    data = []
    for i in range(n_entries):
        claim_id = i + 1
        claim_date = generate_random_date(2010, 2023)
        claim_type = random.choice(['Auto', 'Home', 'Life'])
        claim_severity = random.choice(['Minor', 'Major', 'Catastrophic'])
        claim_amount = random.randint(1000, 100000)
        customer_age = random.randint(18, 70)
        customer_gender = random.choice(['M', 'F'])
        customer_occupation = random.choice(['Professional', 'Blue-collar', 'Unemployed'])
        customer_claims_history = random.randint(0, 5)
        policy_type = random.choice(['Standard', 'Premium'])
        policy_coverage = random.randint(10000, 100000)
        policy_deductible = random.randint(0, 10000)

        row = [claim_id, claim_date, claim_type, claim_severity, claim_amount,
               customer_age, customer_gender, customer_occupation, customer_claims_history,
               policy_type, policy_coverage, policy_deductible]
        data.append(row)

    # Creazione del dataframe
    df = pd.DataFrame(data, columns=columns)

    return df

# Esempio di utilizzo
dataset = generate_claims_dataset(1000)  # Genera un dataset con 1000 righe
print(dataset.info)
dataset.to_csv('claims_data.csv', index=False)
