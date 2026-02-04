import pandas as pd
import numpy as np

np.random.seed(42)  # Para reproducibilidad

def generate_data(n_samples=500):
    data = []
    for _ in range(n_samples):
        diagnostico = np.random.choice(
            ["normal", "gripe", "hipertension", "deshidratacion", "diabetes"],
            p=[0.3, 0.2, 0.15, 0.15, 0.2]  # Distribuciones ajustables
        )
        
        if diagnostico == "normal":
            temp = np.random.normal(36.5, 0.3)
            presion = np.random.randint(90, 121)
            glucosa = np.random.randint(70, 100)
            dolor = 0
            sed = 0
            
        elif diagnostico == "gripe":
            temp = np.random.normal(38.0, 0.5)
            presion = np.random.randint(100, 131)
            glucosa = np.random.randint(70, 100)
            dolor = np.random.randint(1, 4)
            sed = np.random.randint(1, 3)
            
        elif diagnostico == "hipertension":
            temp = np.random.normal(36.8, 0.2)
            presion = np.random.randint(140, 181)
            glucosa = np.random.randint(70, 111)
            dolor = np.random.randint(2, 5)
            sed = np.random.randint(1, 4)
            
        elif diagnostico == "deshidratacion":
            temp = np.random.normal(37.2, 0.4)
            presion = np.random.choice([np.random.randint(85, 100), np.random.randint(100, 121)])
            glucosa = np.random.randint(70, 101)
            dolor = np.random.randint(2, 5)
            sed = np.random.randint(3, 6)
            
        elif diagnostico == "diabetes":
            temp = np.random.normal(36.7, 0.2)
            presion = np.random.randint(110, 151)
            glucosa = np.random.choice([
                np.random.randint(126, 200),  # Ayunas
                np.random.randint(200, 301)   # Aleatorio
            ])
            dolor = np.random.randint(1, 4)
            sed = np.random.randint(4, 6)
            
        data.append([round(temp, 1), presion, glucosa, dolor, sed, diagnostico])
    
    return pd.DataFrame(data, columns=["Temperatura", "Presion", "Glucosa", "DolorCabeza", "Sed", "Diagnostico"])

df = generate_data()
df.to_csv("diagnosticos_dataset.csv", index=False)
