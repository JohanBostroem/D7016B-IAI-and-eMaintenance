#%%
# Email Phishing Detection
# Layer 1: Email Classification
# Layer 2: Multi-Class Threat Classification
# Layer 3: Mitre ATT&CK Classification
# To be added agentic email 

#%% Imports

import os
import random
import sys
import subprocess
import numpy as np
import pandas as pd
import torch

from pathlib import Path

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


#%% Configuration

MODEL_ID = "JustAPR/email-phishing-distilbert-finetuned"

DATA_DIR = Path("data")
RESULTS_DIR = Path("evaluation_results")

# Dataset to process
# Replace dataset by changing dataset here.
DATASET_NAME = "email_dataset_100k.csv"
DATASET_SUBDIR = "Email Classification Data"

# Local copy of the frozen Layer 1 model
MODEL_DIR = Path("models/layer1")
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
MODEL_WEIGHTS_DIR = MODEL_DIR / "model"

BATCH_SIZE = 16

MODEL_LABEL_MAP = {
    0: "legitimate",
    1: "phishing"
}

LAYER1_LABEL_MAP = {
    "legitimate": "legitimate",
    "phishing": "illegitimate"
}


#%% Environment

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Transformers:", __import__("transformers").__version__)


#%% Model setup

# Check whether the frozen Layer 1 model already exists locally
local_model_exists = (
    TOKENIZER_DIR.exists()
    and MODEL_WEIGHTS_DIR.exists()
)

if local_model_exists:
    print("Loading frozen Layer 1 model from local files...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_WEIGHTS_DIR
    )

    model_source = "local"

else:
    print("Local Layer 1 model not found.")
    print(f"Downloading model from Hugging Face: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    # Save a local copy of the frozen Layer 1 model
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(
        TOKENIZER_DIR
    )

    model.save_pretrained(
        MODEL_WEIGHTS_DIR
    )

    model_source = "Hugging Face"

    print(f"Layer 1 tokenizer saved to: {TOKENIZER_DIR}")
    print(f"Layer 1 model saved to: {MODEL_WEIGHTS_DIR}")


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

layer1_model = model
layer1_tokenizer = tokenizer

print("\nLayer 1 model source:", model_source)
print("Device:", device)
print("Labels:", MODEL_LABEL_MAP)


#%% Prediction functions

def classify_email(email):
    """
    Classify a single email.

    Returns:
        label: Predicted class
        confidence: Model confidence
        probabilities: Probability for each class
    """

    inputs = tokenizer(
        email,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    prediction = probabilities.argmax().item()
    confidence = probabilities.max().item()

    return {
        "label": MODEL_LABEL_MAP[prediction],
        "confidence": confidence,
        "probabilities": probabilities.squeeze().tolist()
    }


def classify_emails_batch(emails, batch_size=BATCH_SIZE):
    """
    Classify multiple emails in batches.

    Returns:
        predictions: Predicted class IDs
        confidences: Model confidence for each prediction
    """

    predictions = []
    confidences = []

    for start in range(0, len(emails), batch_size):

        batch = emails[start:start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

        batch_predictions = probabilities.argmax(dim=1)
        batch_confidences = probabilities.max(dim=1).values

        predictions.extend(
            batch_predictions.cpu().tolist()
        )

        confidences.extend(
            batch_confidences.cpu().tolist()
        )

    return predictions, confidences

#%% Email preparation functions

def prepare_email_text(
    df,
    subject_column="subject",
    body_column="body"
):
    """
    Combine subject and body into one text field.
    """

    subject = (
        df[subject_column]
        .fillna("")
        .astype(str)
    )

    body = (
        df[body_column]
        .fillna("")
        .astype(str)
    )

    return subject + " " + body

#%% Dataset setup

DATASET_PATH = DATA_DIR / DATASET_SUBDIR / DATASET_NAME

dataset = pd.read_csv(DATASET_PATH)

dataset_stem = Path(DATASET_NAME).stem

print("Dataset:", DATASET_NAME)
print("Path:", DATASET_PATH)
print("Rows:", len(dataset))
print("Columns:", len(dataset.columns))

#%% Layer 1 - Run or load predictions

layer1_output_dir = RESULTS_DIR / "layer1"
layer1_output_dir.mkdir(parents=True, exist_ok=True)

layer1_output_path = (
    layer1_output_dir /
    f"{dataset_stem}_layer1.csv"
)

required_layer1_columns = {
    "layer1_prediction",
    "layer1_confidence"
}

source_columns = set(dataset.columns)

run_layer1 = True


# Check whether existing Layer 1 predictions match the dataset
if layer1_output_path.exists():

    print("\n" + "=" * 80)
    print("LAYER 1 OUTPUT FOUND")
    print("=" * 80)

    existing_layer1 = pd.read_csv(
        layer1_output_path
    )

    output_is_valid = (
        len(existing_layer1) == len(dataset)
        and source_columns.issubset(
            existing_layer1.columns
        )
        and required_layer1_columns.issubset(
            existing_layer1.columns
        )
    )

    # Verify that the original dataset columns and row order match
    if output_is_valid:

        cached_source = (
            existing_layer1[dataset.columns]
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )

        current_source = (
            dataset
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )

        output_is_valid = cached_source.equals(
            current_source
        )

    if output_is_valid:

        print("Existing Layer 1 predictions match the dataset.")
        print("Loading existing results.")

        layer1_dataset = existing_layer1
        run_layer1 = False

    else:

        print("Existing Layer 1 output is invalid or outdated.")
        print("Layer 1 predictions will be run again.")


# Run Layer 1 only if no valid result exists
if run_layer1:

    print("\n" + "=" * 80)
    print("LAYER 1 - RUNNING PREDICTIONS")
    print("=" * 80)

    # Create a working copy of the original dataset
    layer1_dataset = dataset.copy()

    print(
        "\nCreated Layer 1 working copy."
    )

    print(
        "Original rows:",
        len(dataset)
    )

    print(
        "Working copy rows:",
        len(layer1_dataset)
    )

    # Prepare email text temporarily
    email_texts = prepare_email_text(
        layer1_dataset,
        subject_column="subject",
        body_column="body_plain"
    ).tolist()

    print(
        "\nPrepared",
        len(email_texts),
        "emails for Layer 1."
    )

    # Run model inference
    predictions, confidences = (
        classify_emails_batch(
            email_texts,
            batch_size=BATCH_SIZE
        )
    )

    # Add Layer 1 results
    layer1_dataset[
        "layer1_prediction"
    ] = [
        LAYER1_LABEL_MAP[
            MODEL_LABEL_MAP[prediction]
        ]
        for prediction in predictions
    ]

    layer1_dataset[
        "layer1_confidence"
    ] = confidences

    # Save Layer 1 result
    layer1_dataset.to_csv(
        layer1_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nLayer 1 predictions completed."
    )

    print(
        "Layer 1 results saved to:",
        layer1_output_path
    )


else:

    print(
        "\nLayer 1 inference skipped."
    )


print("\n" + "=" * 80)
print("LAYER 1 COMPLETE")
print("=" * 80)

print(
    "Dataset:",
    DATASET_NAME
)

print(
    "Rows:",
    len(layer1_dataset)
)

print(
    "Layer 1 output:",
    layer1_output_path
)

#%% Test predictions

phishing_email = """
Dear customer,

Your account has been suspended.
Please click the link below to verify your identity immediately.

Thank you,
Security Team
"""

legitimate_email = """
Hi Johan,

Just a reminder that our meeting is scheduled for tomorrow at 10:00.
Please let me know if you need to reschedule.

Best regards,
Anna
"""


print("\n" + "=" * 80)
print("MODEL SANITY TESTS")
print("=" * 80)


result = classify_email(phishing_email)

print("\nPhishing test:")
print("Prediction:", result["label"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Probabilities:", result["probabilities"])


result = classify_email(legitimate_email)

print("\nLegitimate test:")
print("Prediction:", result["label"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Probabilities:", result["probabilities"])

#%% Evaluation helper

def calculate_metrics(true_labels, predictions):

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        true_labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

#%% Prepare unseen Email 100k dataset

print("\n" + "=" * 80)
print("PREPARING EMAIL 100K DATASET")
print("=" * 80)

print("Original shape:", dataset.shape)

# Keep only columns required for evaluation
evaluation_100k = dataset[
    [
        "subject",
        "body_plain",
        "label"
    ]
].copy()

# Handle missing text
evaluation_100k["subject"] = (
    evaluation_100k["subject"]
    .fillna("")
    .astype(str)
)

evaluation_100k["body_plain"] = (
    evaluation_100k["body_plain"]
    .fillna("")
    .astype(str)
)


# Combine subject and body
evaluation_100k["email_text"] = prepare_email_text(
    evaluation_100k,
    subject_column="subject",
    body_column="body_plain"
)


print(
    "After text preparation:",
    evaluation_100k.shape
)

print("\nLabel distribution:")
print(
    evaluation_100k["label"]
    .value_counts()
    .sort_index()
)

print("\nExample prepared email:")
print(
    evaluation_100k.iloc[0]["email_text"][:1000]
)


#%% Evaluate model on unseen Email 100k dataset

print("\n" + "=" * 80)
print("EVALUATING UNSEEN EMAIL 100K DATASET")
print("=" * 80)

# Use the predictions already generated by Layer 1
predictions = layer1_dataset[
    "layer1_prediction"
].tolist()

prediction_ids = (
    layer1_dataset["layer1_prediction"]
    .map({
        "legitimate": 0,
        "illegitimate": 1
    })
    .tolist()
)

confidences = layer1_dataset[
    "layer1_confidence"
].tolist()

true_labels = layer1_dataset[
    "label"
].tolist()

print(
    "\nUsing cached Layer 1 predictions."
)

print(
    "Predictions:",
    len(predictions)
)


print("\nPrediction complete.")
print(
    "Emails evaluated:",
    len(predictions)
)

print(
    "Predicted legitimate:",
    prediction_ids.count(0)
)

print(
    "Predicted illigitimate:",
    prediction_ids.count(1)
)


#%% Calculate unseen dataset metrics

metrics_100k = calculate_metrics(
    true_labels,
    prediction_ids
)

cm = metrics_100k["confusion_matrix"]

tn, fp, fn, tp = cm.ravel()

false_positive_rate = (
    fp / (fp + tn)
)

false_negative_rate = (
    fn / (fn + tp)
)

average_confidence = (
    sum(confidences) /
    len(confidences)
)


print("\n" + "=" * 80)
print("UNSEEN DATASET RESULTS")
print("=" * 80)

print(
    f"Accuracy:           "
    f"{metrics_100k['accuracy']:.2%}"
)

print(
    f"Precision:          "
    f"{metrics_100k['precision']:.2%}"
)

print(
    f"Recall:             "
    f"{metrics_100k['recall']:.2%}"
)

print(
    f"F1-score:           "
    f"{metrics_100k['f1']:.2%}"
)

print(
    f"Average confidence: "
    f"{average_confidence:.2%}"
)

print("\nConfusion matrix:")
print(cm)

print("\nConfusion matrix details:")
print(f"True negatives:      {tn}")
print(f"False positives:     {fp}")
print(f"False negatives:     {fn}")
print(f"True positives:      {tp}")

print("\nError rates:")
print(
    f"False positive rate: "
    f"{false_positive_rate:.2%}"
)

print(
    f"False negative rate: "
    f"{false_negative_rate:.2%}"
)


#%% Save unseen dataset predictions

evaluation_100k_results = (
    evaluation_100k.copy()
)

evaluation_100k_results[
    "predicted_label"
] = prediction_ids

evaluation_100k_results[
    "predicted_class"
] = evaluation_100k_results[
    "predicted_label"
].map(MODEL_LABEL_MAP)

evaluation_100k_results[
    "actual_class"
] = evaluation_100k_results[
    "label"
].map(MODEL_LABEL_MAP)

evaluation_100k_results[
    "confidence"
] = confidences


all_100k_results_file = (
    RESULTS_DIR /
    "Email_100k_predictions.csv"
)

evaluation_100k_results.to_csv(
    all_100k_results_file,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nFull prediction results saved to:"
)

print(all_100k_results_file)


#%% Save misclassified emails from unseen dataset

misclassified_100k = (
    evaluation_100k_results[
        evaluation_100k_results["label"] !=
        evaluation_100k_results["predicted_label"]
    ]
    .copy()
)


print("\n" + "=" * 80)
print("MISCLASSIFIED EMAILS - UNSEEN DATASET")
print("=" * 80)

print(
    "Total misclassified:",
    len(misclassified_100k)
)


false_positives = (
    (misclassified_100k["label"] == 0) &
    (misclassified_100k["predicted_label"] == 1)
).sum()

false_negatives = (
    (misclassified_100k["label"] == 1) &
    (misclassified_100k["predicted_label"] == 0)
).sum()


print("\nFalse positives:")
print(false_positives)

print("\nFalse negatives:")
print(false_negatives)


misclassified_100k_file = (
    RESULTS_DIR /
    "Email_100k_misclassified.csv"
)

misclassified_100k.to_csv(
    misclassified_100k_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved to:")
print(misclassified_100k_file)


#%% Error analysis - prepare metadata


metadata_columns = [
    "language",
    "from_domain",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "has_attachments",
    "has_html",
    "num_urls",
    "num_emails_in_body",
    "num_phone_numbers",
    "contains_tracking_token",
    "x_spam_score"
]


metadata = dataset[
    metadata_columns
].copy()


# Add model predictions
metadata["actual_label"] = (
    evaluation_100k["label"].values
)

metadata["predicted_label"] = prediction_ids

metadata["confidence"] = confidences


# Identify error type
metadata["error_type"] = "correct"

metadata.loc[
    (
        (metadata["actual_label"] == 0) &
        (metadata["predicted_label"] == 1)
    ),
    "error_type"
] = "false_positive"

metadata.loc[
    (
        (metadata["actual_label"] == 1) &
        (metadata["predicted_label"] == 0)
    ),
    "error_type"
] = "false_negative"


print("\n" + "=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)

print("\nError distribution:")
print(
    metadata["error_type"]
    .value_counts()
)


#%% Error analysis by language

print("\n" + "=" * 80)
print("ERRORS BY LANGUAGE")
print("=" * 80)

language_analysis = pd.crosstab(
    metadata["language"],
    metadata["error_type"]
)

print(language_analysis)


#%% Error analysis by email characteristics

print("\n" + "=" * 80)
print("ERRORS BY EMAIL CHARACTERISTICS")
print("=" * 80)


categorical_features = [
    "has_attachments",
    "has_html",
    "contains_tracking_token"
]


for column in categorical_features:

    print(f"\n--- {column} ---")

    print(
        pd.crosstab(
            metadata[column],
            metadata["error_type"]
        )
    )


#%% Error analysis - numerical features

print("\n" + "=" * 80)
print("NUMERICAL FEATURE COMPARISON")
print("=" * 80)


numerical_features = [
    "num_urls",
    "num_emails_in_body",
    "num_phone_numbers",
    "x_spam_score"
]


for column in numerical_features:

    print(f"\n--- {column} ---")

    print(
        metadata
        .groupby("error_type")[column]
        .agg([
            "count",
            "mean",
            "median"
        ])
    )


#%% Error rates by categorical features

print("\n" + "=" * 80)
print("ERROR RATES BY EMAIL CHARACTERISTICS")
print("=" * 80)


for column in [
    "language",
    "has_attachments",
    "has_html",
    "contains_tracking_token"
]:

    print(f"\n--- {column} ---")

    for value in sorted(
        metadata[column]
        .dropna()
        .unique()
    ):

        subset = metadata[
            metadata[column] == value
        ]

        malicious = subset[
            subset["actual_label"] == 1
        ]

        legitimate = subset[
            subset["actual_label"] == 0
        ]

        false_negative_rate = (
            (
                malicious["predicted_label"] == 0
            ).mean()
            if len(malicious) > 0
            else 0
        )

        false_positive_rate = (
            (
                legitimate["predicted_label"] == 1
            ).mean()
            if len(legitimate) > 0
            else 0
        )

        print(
            f"{value}: "
            f"FN rate = {false_negative_rate:.2%}, "
            f"FP rate = {false_positive_rate:.2%}, "
            f"n = {len(subset)}"
        )

#%%
# =============================================================================
# LAYER 2: BERT MULTI-CLASS EMAIL THREAT CLASSIFICATION
# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPSILON = 1e-8
EPOCHS = 4
WARMUP_STEPS = 0
GRADIENT_CLIP = 1.0

DATA_PATH = "data/Layer 2 dataset/paper_ready_dataset.csv"

OUTPUT_DIR = "evaluation_results/layer2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = Path("models/layer2")
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
MODEL_WEIGHTS_DIR = MODEL_DIR / "model"

# =============================================================================
# END-TO-END PIPELINE EVALUATION
# =============================================================================

PIPELINE_DATA_PATH = "data/Pipeline dataset/Pipeline_test_100.csv"

PIPELINE_RESULTS_DIR = Path(
    "evaluation_results/pipeline"
)

PIPELINE_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =============================================================================
# TRAINING CONTROL
# =============================================================================

FORCE_RETRAIN = False

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DEVICE
# =============================================================================

if torch.cuda.is_available():
    device = torch.device("cuda")

    print(f"GPU available.")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

print(f"Device: {device}")

# =============================================================================
# LOAD DATASET
# =============================================================================

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {DATA_PATH}")
print(f"Number of emails: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["class"].value_counts())

# =============================================================================
# PREPARE EMAIL TEXT
# =============================================================================

df = df.dropna(subset=["class"]).copy()

# Combine subject and body.
df["bert_text"] = (
    df["subject"].fillna("").astype(str)
    + " "
    + df["body"].fillna("").astype(str)
)

# Remove rows where both subject and body are empty.
df = df[df["bert_text"].str.strip() != ""].copy()

print(f"Emails available for Layer 2: {len(df)}")

print("\nExample:")
print(df[["subject", "body", "class", "bert_text"]].head(1).to_string(index=False))

# =============================================================================
# LABEL ENCODING AND DATA SPLITTING
# =============================================================================

le = LabelEncoder()

df["class_encoded"] = le.fit_transform(df["class"])

class_names = le.classes_
num_classes = len(class_names)

print(f"Number of classes: {num_classes}")
print("\nClass mapping:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# Features and target
X = df["bert_text"]
y = df["class_encoded"]


# 75/25 stratified split, matching the paper
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=SEED,
    stratify=y
)


print("\nData split:")
print(f"Training samples: {len(X_train_text)}")
print(f"Test samples:     {len(X_test_text)}")
print(f"Total samples:    {len(df)}")

# =============================================================================
# BERT TOKENIZATION
# =============================================================================

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def bert_encode(texts, max_len=MAX_LENGTH):
    return tokenizer(
        texts.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )


print("Tokenizing training data...")
encoded_train = bert_encode(X_train_text)

print("Tokenizing test data...")
encoded_test = bert_encode(X_test_text)

print("Tokenization complete.")

print(f"Train tensor shape: {encoded_train['input_ids'].shape}")
print(f"Test tensor shape:  {encoded_test['input_ids'].shape}")

# =============================================================================
# PYTORCH DATASETS AND DATALOADERS
# =============================================================================

train_inputs = encoded_train["input_ids"]
train_masks = encoded_train["attention_mask"]
train_labels = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.long
)

test_inputs = encoded_test["input_ids"]
test_masks = encoded_test["attention_mask"]
test_labels = torch.tensor(
    y_test.to_numpy(),
    dtype=torch.long
)


# Training dataset
train_data = TensorDataset(
    train_inputs,
    train_masks,
    train_labels
)

g = torch.Generator()
g.manual_seed(SEED)

train_sampler = RandomSampler(
    train_data,
    generator=g
)

train_dataloader = DataLoader(
    train_data,
    sampler=train_sampler,
    batch_size=BATCH_SIZE
)


# Test dataset
test_data = TensorDataset(
    test_inputs,
    test_masks,
    test_labels
)

test_sampler = SequentialSampler(test_data)

test_dataloader = DataLoader(
    test_data,
    sampler=test_sampler,
    batch_size=BATCH_SIZE
)


print(f"Training batches: {len(train_dataloader)}")
print(f"Test batches:     {len(test_dataloader)}")


# =============================================================================
# BERT MODEL
# =============================================================================

model_exists = (
    (TOKENIZER_DIR / "tokenizer.json").exists()
    and
    (MODEL_WEIGHTS_DIR / "config.json").exists()
    and
    (MODEL_WEIGHTS_DIR / "model.safetensors").exists()
)


if FORCE_RETRAIN or not model_exists:

    # -------------------------------------------------------------------------
    # TRAIN A NEW MODEL
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING NEW LAYER 2 BERT MODEL")
    print("=" * 70)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False
    )

    model.to(device)

    print(f"Model: {MODEL_NAME}")
    print(f"Number of output classes: {num_classes}")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # OPTIMIZER AND LEARNING RATE SCHEDULER
    # -------------------------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        eps=EPSILON
    )

    total_steps = len(train_dataloader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    print(f"Total training steps: {total_steps}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    training_history = []

    for epoch in range(EPOCHS):

        print("")
        print("=" * 70)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 70)

        model.train()

        total_train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear previous gradients
            model.zero_grad()

            # Forward pass
            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
                return_dict=True
            )

            loss = result.loss

            total_train_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            # Update model weights
            optimizer.step()

            # Update learning rate
            scheduler.step()

            # Progress
            if (step + 1) % 100 == 0:
                print(
                    f"Step {step + 1}/{len(train_dataloader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        avg_train_loss = (
            total_train_loss / len(train_dataloader)
        )

        training_history.append({
            "epoch": epoch + 1,
            "training_loss": avg_train_loss
        })

        print(
            f"\nAverage training loss: "
            f"{avg_train_loss:.4f}"
        )

    print("\nTraining complete.")

    # -------------------------------------------------------------------------
    # SAVE TRAINED MODEL
    # -------------------------------------------------------------------------

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_pretrained(
        MODEL_WEIGHTS_DIR
    )

    tokenizer.save_pretrained(
        TOKENIZER_DIR
    )

    print("\nLayer 2 model saved.")
    print(f"Model:     {MODEL_WEIGHTS_DIR}")
    print(f"Tokenizer: {TOKENIZER_DIR}")


else:

    # -------------------------------------------------------------------------
    # LOAD EXISTING MODEL
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING EXISTING LAYER 2 BERT MODEL")
    print("=" * 70)

    print(f"Model location: {MODEL_WEIGHTS_DIR}")
    print(f"Tokenizer location: {TOKENIZER_DIR}")

    model = BertForSequenceClassification.from_pretrained(
        MODEL_WEIGHTS_DIR
    )

    tokenizer = BertTokenizer.from_pretrained(
        TOKENIZER_DIR
    )

    model.to(device)

    print("Existing Layer 2 model loaded.")
    print(f"Device: {device}")

# Keep explicit references to the Layer 2 model
layer2_model = model
layer2_tokenizer = tokenizer


# =============================================================================
# EVALUATION
# =============================================================================

model.eval()

all_preds = []
all_labels = []
all_probas = []


with torch.no_grad():

    for batch in test_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        result = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            return_dict=True
        )

        logits = result.logits

        # Convert logits to class probabilities
        probas = torch.nn.functional.softmax(
            logits,
            dim=-1
        ).cpu().numpy()

        # Predicted class
        preds = np.argmax(
            probas,
            axis=1
        )

        all_preds.extend(preds)
        all_labels.extend(
            b_labels.cpu().numpy()
        )
        all_probas.extend(probas)


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probas = np.array(all_probas)

print("Evaluation complete.")
print(f"Predictions: {len(all_preds)}")
print(f"Probability shape: {all_probas.shape}")


# =============================================================================
# METRICS
# =============================================================================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average="macro"
)

macro_recall = recall_score(
    all_labels,
    all_preds,
    average="macro"
)

# One-vs-rest multiclass ROC-AUC
try:
    macro_auc = roc_auc_score(
        all_labels,
        all_probas,
        multi_class="ovr",
        average="macro"
    )
except ValueError:
    macro_auc = np.nan


print("\n" + "=" * 70)
print("LAYER 2 RESULTS")
print("=" * 70)

print(f"Accuracy:      {accuracy:.4f}")
print(f"Macro F1:      {macro_f1:.4f}")
print(f"Macro Recall:  {macro_recall:.4f}")
print(f"Macro ROC-AUC: {macro_auc:.4f}")


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

print("\nConfusion Matrix:")
print(cm_df)


# =============================================================================
# MISCLASSIFIED EMAILS
# =============================================================================

test_results = df.loc[
    X_test_text.index
].copy()

test_results["true_class"] = [
    class_names[label]
    for label in all_labels
]

test_results["predicted_class"] = [
    class_names[pred]
    for pred in all_preds
]

test_results["confidence"] = np.max(
    all_probas,
    axis=1
)

misclassified = test_results[
    test_results["true_class"] != test_results["predicted_class"]
].copy()

misclassified = misclassified.sort_values(
    "confidence",
    ascending=False
)

misclassified_path = os.path.join(
    OUTPUT_DIR,
    "layer2_misclassified.csv"
)

misclassified.to_csv(
    misclassified_path,
    index=False
)

print(f"Misclassified emails: {len(misclassified)}")
print(f"Saved to: {misclassified_path}")

print("\nExamples:")
print(
    misclassified[
        [
            "subject",
            "true_class",
            "predicted_class",
            "confidence"
        ]
    ].head(20).to_string(index=False)
)


# =============================================================================
# SAVE LAYER 2 PREDICTIONS
# =============================================================================

layer2_results = df.loc[
    X_test_text.index
].copy()

layer2_results["true_class"] = [
    class_names[label]
    for label in all_labels
]

layer2_results["predicted_class"] = [
    class_names[pred]
    for pred in all_preds
]

layer2_results["confidence"] = np.max(
    all_probas,
    axis=1
)


# Add probability for every class
for class_index, class_name in enumerate(class_names):
    safe_name = (
        class_name
        .lower()
        .replace(" ", "_")
    )

    layer2_results[
        f"prob_{safe_name}"
    ] = all_probas[:, class_index]


layer2_output_path = os.path.join(
    OUTPUT_DIR,
    "layer2_predictions.csv"
)

layer2_results.to_csv(
    layer2_output_path,
    index=False
)

print(f"Layer 2 predictions saved to:")
print(layer2_output_path)


#%%
# =============================================================================
# END-TO-END PIPELINE: LAYER 1 -> LAYER 2 -> LAYER 3
# =============================================================================

def classify_emails_batch_layer1(
    emails,
    batch_size=BATCH_SIZE
):
    """Classify emails using the frozen Layer 1 model."""
    predictions = []
    confidences = []

    layer1_model.eval()

    for start in range(0, len(emails), batch_size):
        batch = emails[start:start + batch_size]

        inputs = layer1_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = layer1_model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

        predictions.extend(
            probabilities.argmax(dim=1).cpu().tolist()
        )

        confidences.extend(
            probabilities.max(dim=1).values.cpu().tolist()
        )

    return predictions, confidences


print("\n" + "=" * 80)
print("END-TO-END PIPELINE: LAYER 1 -> LAYER 2 -> LAYER 3")
print("=" * 80)

# Resolve paths relative to the directory containing this script.
project_root = Path(__file__).resolve().parent

pipeline_data_path = (
    project_root / PIPELINE_DATA_PATH
)

pipeline_results_dir = (
    project_root / PIPELINE_RESULTS_DIR
)

pipeline_results_dir.mkdir(
    parents=True,
    exist_ok=True
)

# -------------------------------------------------------------------------
# Load the 100-email test dataset
# -------------------------------------------------------------------------

print("\nLoading pipeline test dataset:")
print(pipeline_data_path)

if not pipeline_data_path.is_file():
    raise FileNotFoundError(
        f"Pipeline dataset not found: {pipeline_data_path}"
    )

pipeline_df = pd.read_csv(pipeline_data_path)

required_pipeline_columns = {
    "id",
    "subject",
    "body",
    "attachment",
    "contains_attachment",
}

missing_columns = (
    required_pipeline_columns - set(pipeline_df.columns)
)

if missing_columns:
    raise ValueError(
        "Pipeline dataset is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

if pipeline_df.empty:
    raise ValueError("Pipeline dataset contains no emails.")

if pipeline_df["id"].isna().any():
    raise ValueError("Every pipeline email must have an ID.")

if pipeline_df["id"].astype(str).duplicated().any():
    raise ValueError("Pipeline dataset contains duplicate IDs.")

pipeline_df = pipeline_df.reset_index(drop=True)

print(f"Emails loaded: {len(pipeline_df)}")

# -------------------------------------------------------------------------
# Layer 1: classify ALL emails as legitimate or illegitimate
# -------------------------------------------------------------------------

print("\n" + "=" * 80)
print("LAYER 1: BINARY CLASSIFICATION")
print("=" * 80)

layer1_texts = prepare_email_text(
    pipeline_df,
    subject_column="subject",
    body_column="body"
).tolist()

layer1_predictions, layer1_confidences = (
    classify_emails_batch_layer1(
        layer1_texts,
        batch_size=BATCH_SIZE
    )
)

pipeline_df["layer1_prediction"] = [
    LAYER1_LABEL_MAP[
        MODEL_LABEL_MAP[int(prediction)]
    ]
    for prediction in layer1_predictions
]

pipeline_df["layer1_confidence"] = layer1_confidences

print("\nLayer 1 prediction distribution:")
print(pipeline_df["layer1_prediction"].value_counts())

# -------------------------------------------------------------------------
# Prepare Layer 2 result columns
# -------------------------------------------------------------------------

pipeline_df["layer2_predicted_class"] = ""
pipeline_df["layer2_confidence"] = np.nan

pipeline_df["pipeline_status"] = (
    "stopped_after_layer1_legitimate"
)

# Only emails predicted as illegitimate proceed to Layer 2.
eligible_indices = pipeline_df.index[
    pipeline_df["layer1_prediction"] == "illegitimate"
].tolist()

pipeline_df.loc[
    eligible_indices,
    "pipeline_status"
] = "routed_to_layer2"

print(
    "\nEmails routed from Layer 1 to Layer 2:",
    len(eligible_indices)
)

# -------------------------------------------------------------------------
# Layer 2: classify eligible emails using the saved BERT model
# -------------------------------------------------------------------------

print("\n" + "=" * 80)
print("LAYER 2: MULTI-CLASS CLASSIFICATION")
print("=" * 80)

if layer2_model.config.num_labels != len(class_names):
    raise ValueError(
        "The Layer 2 model output count does not match class_names. "
        "Check the saved model and its class mapping."
    )

layer2_model.eval()

layer2_predicted_classes = []
layer2_prediction_confidences = []

if eligible_indices:

    eligible_df = pipeline_df.loc[eligible_indices]

    layer2_texts = prepare_email_text(
        eligible_df,
        subject_column="subject",
        body_column="body"
    ).tolist()

    with torch.no_grad():

        for start in range(0, len(layer2_texts), BATCH_SIZE):

            batch_texts = layer2_texts[
                start:start + BATCH_SIZE
            ]

            inputs = layer2_tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH
            ).to(device)

            outputs = layer2_model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

            predicted_ids = probabilities.argmax(
                dim=1
            ).cpu().tolist()

            confidences = probabilities.max(
                dim=1
            ).values.cpu().tolist()

            layer2_predicted_classes.extend(
                str(class_names[int(predicted_id)])
                for predicted_id in predicted_ids
            )

            layer2_prediction_confidences.extend(
                confidences
            )

    pipeline_df.loc[
        eligible_indices,
        "layer2_predicted_class"
    ] = layer2_predicted_classes

    pipeline_df.loc[
        eligible_indices,
        "layer2_confidence"
    ] = layer2_prediction_confidences

    pipeline_df.loc[
        eligible_indices,
        "pipeline_status"
    ] = "ready_for_layer3"

print("\nLayer 2 prediction distribution:")

if eligible_indices:
    print(
        pipeline_df.loc[
            eligible_indices,
            "layer2_predicted_class"
        ].value_counts()
    )
else:
    print("No emails were routed to Layer 2.")

# -------------------------------------------------------------------------
# Save complete Layer 1 + Layer 2 results for all original emails
# -------------------------------------------------------------------------

pipeline_results_path = (
    pipeline_results_dir
    / "pipeline_100_layer1_layer2_results.csv"
)

pipeline_df.to_csv(
    pipeline_results_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nComplete Layer 1 + Layer 2 results saved:")
print(pipeline_results_path)

# -------------------------------------------------------------------------
# Prepare input for Layer 3
# Include only emails that proceeded through Layers 1 and 2.
# -------------------------------------------------------------------------

layer3_input = pipeline_df.loc[
    eligible_indices
].copy()

# Use these exact field names expected by layer3.py.
layer3_input["predicted_class"] = (
    layer3_input["layer2_predicted_class"]
)

layer3_input["confidence"] = (
    layer3_input["layer2_confidence"]
)

layer3_input_path = (
    pipeline_results_dir
    / "pipeline_layer2_predictions.csv"
)

layer3_input.to_csv(
    layer3_input_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nLayer 3 input saved:")
print(layer3_input_path)

print("\n" + "=" * 80)
print("LAYER 1 + LAYER 2 PIPELINE SUMMARY")
print("=" * 80)

print(f"Original emails: {len(pipeline_df)}")
print(
    "Stopped after Layer 1:",
    len(pipeline_df) - len(eligible_indices)
)
print("Passed to Layer 2:", len(eligible_indices))
print("Prepared for Layer 3:", len(layer3_input))

# -------------------------------------------------------------------------
# Automatically launch Layer 3
# -------------------------------------------------------------------------

if layer3_input.empty:

    print(
        "\nWARNING: No emails passed Layer 1. "
        "Layer 3 will not be started."
    )

else:

    print("\n" + "=" * 80)
    print("STARTING LAYER 3")
    print("=" * 80)

    layer3_script = (
        project_root
        / "models"
        / "layer3"
        / "layer3.py"
    )

    if not layer3_script.is_file():
        raise FileNotFoundError(
            f"Layer 3 script not found: {layer3_script}"
        )

    print("Layer 3 script:", layer3_script)
    print("Layer 3 input:", layer3_input_path)
    print("Starting Layer 3 with the current Python environment...")

    # This waits until Layer 3 finishes.
    # VirusTotal lookups remain enabled in layer3.py.
    subprocess.run(
        [sys.executable, str(layer3_script)],
        cwd=project_root,
        check=True
    )

    print("\n" + "=" * 80)
    print("ALL THREE LAYERS FINISHED")
    print("=" * 80)
