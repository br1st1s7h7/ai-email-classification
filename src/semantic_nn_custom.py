
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

emails = []
labels = []

for file_path in DATA_DIR.glob("*.json"):
    category = file_path.stem.split("_")[0].upper()

    with open(file_path, "r", encoding="utf-8") as file:
        batch = json.load(file)

    for email in batch:
        subject = email.get("subject") or ""
        body = email.get("body") or ""

        emails.append(subject + " " + body)
        labels.append(category)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    emails,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Loading semantic embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

X_train_embeddings = embedding_model.encode(
    X_train,
    show_progress_bar=True
)

X_test_embeddings = embedding_model.encode(
    X_test,
    show_progress_bar=True
)

tf.keras.utils.set_random_seed(42)


class EmailClassifier(Model):
    def __init__(self, num_classes):
        super().__init__()

        self.dense1 = Dense(
            128,
            activation="relu",
            kernel_regularizer=l2(0.001)
        )
        self.dropout1 = Dropout(0.3)
        self.dense2 = Dense(64, activation="relu")
        self.dropout2 = Dropout(0.2)
        self.output_layer = Dense(
            num_classes,
            activation="softmax"
        )

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)


model = EmailClassifier(len(label_encoder.classes_))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

model.fit(
    X_train_embeddings,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

probabilities = model.predict(X_test_embeddings)
predictions = np.argmax(probabilities, axis=1)

print("\nTotal emails:", len(emails))
print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))
