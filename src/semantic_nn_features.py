
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

subjects = []
bodies = []
attachments = []
numeric_features = []
labels = []

for file_path in DATA_DIR.glob("*.json"):
    category = file_path.stem.split("_")[0].upper()

    with open(file_path, "r", encoding="utf-8") as file:
        batch = json.load(file)

    for email in batch:
        subject = email.get("subject") or ""
        body = email.get("body") or ""
        attachment_names = email.get("attachment_names") or []

        if isinstance(attachment_names, str):
            attachment_names = [attachment_names]

        attachment_text = " ".join(attachment_names)

        subjects.append(subject)
        bodies.append(body)
        attachments.append(attachment_text)

        numeric_features.append([
            len(subject),
            len(body),
            len(attachment_names),
            int(bool(attachment_names))
        ])

        labels.append(category)

subjects = np.array(subjects)
bodies = np.array(bodies)
attachments = np.array(attachments)
numeric_features = np.array(numeric_features, dtype=np.float32)
labels = np.array(labels)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

indices = np.arange(len(labels))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Loading semantic embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

subject_embeddings = embedding_model.encode(
    subjects.tolist(),
    show_progress_bar=True
)

body_embeddings = embedding_model.encode(
    bodies.tolist(),
    show_progress_bar=True
)

attachment_embeddings = embedding_model.encode(
    attachments.tolist(),
    show_progress_bar=True
)

scaler = StandardScaler()

X_numeric_train = scaler.fit_transform(
    numeric_features[train_indices]
)

X_numeric_test = scaler.transform(
    numeric_features[test_indices]
)

X_subject_train = subject_embeddings[train_indices]
X_subject_test = subject_embeddings[test_indices]

X_body_train = body_embeddings[train_indices]
X_body_test = body_embeddings[test_indices]

X_attachment_train = attachment_embeddings[train_indices]
X_attachment_test = attachment_embeddings[test_indices]

y_train = tf.keras.utils.to_categorical(
    y[train_indices], num_classes=len(label_encoder.classes_)
)

y_test = y[test_indices]

subject_input = Input(shape=(384,), name="subject_embedding")
body_input = Input(shape=(384,), name="body_embedding")
attachment_input = Input(shape=(384,), name="attachment_embedding")
numeric_input = Input(shape=(4,), name="numeric_features")

combined = Concatenate()([
    subject_input,
    body_input,
    attachment_input,
    numeric_input
])

x = Dense(256, activation="relu")(combined)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)

output = Dense(
    len(label_encoder.classes_),
    activation="softmax"
)(x)

model = Model(
    inputs=[
        subject_input,
        body_input,
        attachment_input,
        numeric_input
    ],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    [
        X_subject_train,
        X_body_train,
        X_attachment_train,
        X_numeric_train
    ],
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

predicted_probabilities = model.predict([
    X_subject_test,
    X_body_test,
    X_attachment_test,
    X_numeric_test
])

predictions = np.argmax(predicted_probabilities, axis=1)

print("\nTotal emails:", len(labels))
print("Training emails:", len(train_indices))
print("Test emails:", len(test_indices))
print("Embedding dimensions per text input: 384")
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
