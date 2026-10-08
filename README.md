# AraBERT Arabic Sentiment Analysis

Fine-tuning and model compression of **AraBERT** for 3-class Arabic sentiment classification.

The project starts by reducing the original AraBERT architecture from 12 Transformer encoder layers to 6 layers. The resulting lightweight model is then fine-tuned on the Arabic sentiment dataset.

This approach keeps the reduced architecture throughout the training and deployment pipeline.

---

# 1. Installation

Install the required packages:

```bash
pip install datasets pandas scikit-learn torch transformers
```

For the later ONNX optimization stage:

```bash
pip install -U "optimum-onnx[onnxruntime]"
```

---

# 2. Load the AraBERT Model

The base model used in the project is:

```text
aubmindlab/bert-base-arabertv02
```

Load the model as a sequence classification model:

```python
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)
```

At this point, the original AraBERT architecture contains:

```text
12 Transformer Encoder Layers
```

---

# 3. Model Compression

Before starting the training process, the model architecture is reduced from 12 Transformer encoder layers to 6 layers.

The first six encoder layers are kept:

```python
import torch

model.bert.encoder.layer = torch.nn.ModuleList(
    list(model.bert.encoder.layer[:6])
)
```

The model configuration is then updated:

```python
model.config.num_hidden_layers = 6
```

The architecture changes from:

```text
Original AraBERT

Layer 1
Layer 2
Layer 3
Layer 4
Layer 5
Layer 6
Layer 7
Layer 8
Layer 9
Layer 10
Layer 11
Layer 12
```

to:

```text
Compressed AraBERT

Layer 1
Layer 2
Layer 3
Layer 4
Layer 5
Layer 6
```

This reduces the depth of the Transformer before fine-tuning.

### Why compress before fine-tuning?

The goal is to train the smaller architecture directly rather than fine-tuning the full 12-layer model and removing layers afterward.

This means the model that goes through the training pipeline is already the 6-layer version.

---

# 4. Save the Compressed Base Model

The 6-layer model and tokenizer are saved:

```python
OUTPUT_PATH = "/content/models"

model.save_pretrained(
    OUTPUT_PATH
)

tokenizer.save_pretrained(
    OUTPUT_PATH
)
```

The model can then be loaded again:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "./models"
)
```

Verify the architecture:

```python
print(model.config.num_hidden_layers)
print(len(model.bert.encoder.layer))
```

Expected output:

```text
6
6
```

At this point, the project has a compressed 6-layer AraBERT model ready for fine-tuning.

---

# 5. Load the Dataset

The Arabic sentiment dataset is loaded using Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset(
    "akhooli/ar_reviews_100k_3"
)
```

Only the required columns are selected:

```python
dataset = dataset["train"].select_columns(
    ["text", "label"]
)
```

The dataset is exported to:

```text
data/arabic_sentiment.csv
```

```python
dataset.to_csv(
    "data/arabic_sentiment.csv",
    index=False
)
```

---

# 6. Prepare the Data

Load the CSV:

```python
import pandas as pd

df = pd.read_csv(
    "data/arabic_sentiment.csv"
)
```

Remove missing values:

```python
df = df.dropna()
```

Convert the columns to the required types:

```python
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)
```

Check the label distribution:

```python
print(
    df["label"]
    .value_counts()
    .sort_index()
)
```

---

# 7. Split the Dataset

The data is divided into:

```text
80% Training
10% Validation
10% Test
```

First, 20% is separated:

```python
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)
```

Then the temporary 20% is split equally:

```python
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)
```

`stratify` keeps the class distribution consistent across the datasets.

---

# 8. Load the Tokenizer

The tokenizer comes from the original AraBERT model:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

Tokenization is performed with:

```python
def tokenize(data):

    return tokenizer(
        data["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=128
    )
```

The maximum sequence length is 128 tokens.

---

# 9. Create PyTorch Datasets

The tokenized data is wrapped in a custom PyTorch Dataset:

```python
class SentimentDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item
```

Three datasets are created:

```python
train_dataset = SentimentDataset(
    train_tokens,
    train_df["label"]
)

val_dataset = SentimentDataset(
    val_tokens,
    val_df["label"]
)

test_dataset = SentimentDataset(
    test_tokens,
    test_df["label"]
)
```

---

# 10. Fine-Tune the 6-Layer AraBERT

The model used for training is now the compressed 6-layer model.

The training configuration:

```python
training_args = TrainingArguments(

    output_dir="outputs/arabert-sentiment",

    num_train_epochs=3,

    per_device_train_batch_size=9,
    per_device_eval_batch_size=9,

    learning_rate=2e-5,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="loss",

    logging_steps=100,

    report_to="none",

    fp16=True
)
```

The model is trained using:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

The important point is:

```text
12-layer AraBERT
        ↓
6-layer compression
        ↓
6-layer AraBERT
        ↓
Fine-Tuning
```

The model is **not trained as a 12-layer model first**.

---

# 11. Validation During Training

Validation is performed after every epoch because:

```python
eval_strategy="epoch"
```

The following metrics are calculated:

```text
Accuracy
Weighted Precision
Weighted Recall
Weighted F1
```

The best checkpoint is selected based on the lowest validation loss:

```python
load_best_model_at_end=True
metric_for_best_model="loss"
```

---

# 12. Final Test Evaluation

After training, the final model is evaluated on the test set:

```python
result = trainer.predict(
    test_dataset
)
```

Predictions are extracted:

```python
y_true = result.label_ids

y_pred = np.argmax(
    result.predictions,
    axis=1
)
```

Final metrics:

```python
accuracy = accuracy_score(
    y_true,
    y_pred
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
)
```

The results are printed:

```python
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
```

---

# 13. Save the Fine-Tuned 6-Layer Model

After fine-tuning, the final model is saved:

```python
MODEL_PATH = "outputs/arabert-sentiment"

trainer.save_model(
    MODEL_PATH
)

tokenizer.save_pretrained(
    MODEL_PATH
)
```

The saved model is now:

```text
Fine-Tuned
      +
6 Transformer Layers
      +
Arabic Sentiment Classification Head
```

This is the model that will be used for the deployment stage.

---

# 14. ONNX Optimization

The ONNX Runtime optimization package is installed:

```bash
pip install -U "optimum-onnx[onnxruntime]"
```

The next stage is to export the fine-tuned 6-layer AraBERT model to ONNX and use ONNX Runtime for optimized inference.

---

# Final Pipeline

```text
                    AraBERT
                       │
                       ▼
              12 Encoder Layers
                       │
                       ▼
               Layer Reduction
                       │
                       ▼
               6 Encoder Layers
                       │
                       ▼
              Arabic Dataset
                       │
                       ▼
             Data Preparation
                       │
                       ▼
              80 / 10 / 10 Split
                       │
                       ▼
                  Tokenization
                       │
                       ▼
                Fine-Tuning
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Validation            Training
        Every Epoch           3 Epochs
             │                   │
             └─────────┬─────────┘
                       ▼
                Best Checkpoint
                       │
                       ▼
                Final Test Set
                       │
                       ▼
            Accuracy / Precision
             Recall / F1 Score
                       │
                       ▼
              Save Final Model
                       │
                       ▼
                  ONNX Export
                       │
                       ▼
                  Deployment
``` 