import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report

def evaluate():
    model = tf.keras.models.load_model('models/cnn_v1.h5')
    df = pd.read_csv('data/processed/gossipcop_cleaned.csv').sample(500) # Test on a sample
    # ... (Add same tokenization as train.py) ...
    # predictions = (model.predict(X_test) > 0.5).astype("int32")
    # print(classification_report(y_test, predictions))

if __name__ == "__main__":
    print("Evaluation logic ready.")