import pandas as pd

# El script usa el mapeo de labels y en base a este convierte los resultados en formato kaggle, pero
#parece que al hacer esto se alteran los resultados
test_predictions_df = pd.read_csv('test_predictions.csv')
label_mapping_df = pd.read_csv('mapeo_de_labels.csv')
label_to_cluster = dict(zip(label_mapping_df['label'], label_mapping_df['cluster']))
test_predictions_df['label'] = test_predictions_df['label'].map(label_to_cluster)
test_predictions_df.to_csv('sumision.csv', index=False)
print("El archivo 'sumision.csv' ha sido creado con éxito.")
