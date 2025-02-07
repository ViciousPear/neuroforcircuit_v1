import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def create_matrix():
    with open('predictions.json') as js:
        results = json.load(js)

    print(results[:5])
   
    true_labels = [x['category_id'] for x in results]
    pred_labels = [x['category_id'] for x in results]

    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=range(len(set(true_labels))), columns=range(len(set(true_labels))))

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Confusion Matrix')
    plt.show()

create_matrix()