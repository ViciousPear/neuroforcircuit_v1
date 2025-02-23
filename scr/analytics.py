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

def create_schedule_of_losses():
    log_path = "./runs/detect/circuit_elements/results.csv"
    df = pd.read_csv(log_path)

    df["train/loss"] = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
    df["val/loss"] = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]

    # Строим график
    plt.figure(figsize=(20, 10))
    plt.plot(df["epoch"], df["train/loss"], label="Training Loss", marker="o")
    plt.plot(df["epoch"], df["val/loss"], label="Validation Loss", marker="s")

    # Подписываем оси
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

create_schedule_of_losses()