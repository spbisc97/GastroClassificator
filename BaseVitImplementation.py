
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import logging
from torch import nn # if you want to use the nn module for custom classificaiton head
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn import metrics as mtc
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer

# Setting up the logger
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Dataset directories
train_root_dir = "./DATASET/TRAIN"
val_root_dir = "./DATASET/VAL"
test_root_dir = "./DATASET/TEST"

# Load the feature extractor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Function to load datasets
def load_dataset(train_root_dir, val_root_dir, test_root_dir, transform=None):
    train_dataset = datasets.ImageFolder(train_root_dir, transform=transform)
    logger.info("Training dataset loaded with %d samples", len(train_dataset))
    val_dataset = datasets.ImageFolder(val_root_dir, transform=transform)
    logger.info("Validation dataset loaded with %d samples", len(val_dataset))
    test_dataset = datasets.ImageFolder(test_root_dir, transform=transform)
    logger.info("Test dataset loaded with %d samples", len(test_dataset))
    return train_dataset, val_dataset, test_dataset

# Function to get labels from dataset
def get_labels(dataset):
    return dataset.classes

# Function to calculate metrics
def calculate_metrics(outputs, targets):
    metrics = {
        "micro_precision": mtc.precision_score(outputs, targets, average="micro", zero_division=0),
        "micro_recall": mtc.recall_score(outputs, targets, average="micro", zero_division=0),
        "micro_f1": mtc.f1_score(outputs, targets, average="micro", zero_division=0),
        "macro_precision": mtc.precision_score(outputs, targets, average="macro", zero_division=0),
        "macro_recall": mtc.recall_score(outputs, targets, average="macro", zero_division=0),
        "macro_f1": mtc.f1_score(outputs, targets, average="macro", zero_division=0),
        "mcc": mtc.matthews_corrcoef(outputs, targets)
    }
    return metrics

# Function to compute evaluation metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return calculate_metrics(preds, labels)

# Training arguments
metric_name = "micro_f1"
args = TrainingArguments(
    "test-GastroVision",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='./logs',
    logging_steps=30,  # Log every 10 steps
    fp16=True,  # Enable mixed precision training
)

# Load datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_dataset, val_dataset, test_dataset = load_dataset(train_root_dir, val_root_dir, test_root_dir, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Load the model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(get_labels(train_dataset)))

# Move the model to the appropriate device
model.to(device)

# Custom collate function to handle the DataLoader output
def custom_collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# Wrap DataLoader to be compatible with Trainer
class WrappedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield custom_collate_fn(batch)

    def __len__(self):
        return len(self.dataloader)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    data_collator=custom_collate_fn,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
test_results = trainer.evaluate(eval_dataset=test_loader.dataset)
logger.info(f"Test results: {test_results}")

# Generate predictions for the test dataset
preds_output = trainer.predict(test_loader.dataset)
preds = np.argmax(preds_output.predictions, axis=1)
labels = preds_output.label_ids

# Calculate and log the confusion matrix
conf_matrix = confusion_matrix(labels, preds)
logger.info(f"Confusion Matrix:\n{conf_matrix}")

# Plot and save the confusion matrix with annotations
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(get_labels(test_dataset)))
plt.xticks(tick_marks, get_labels(test_dataset), rotation=45)
plt.yticks(tick_marks, get_labels(test_dataset))

# Annotate each cell with the corresponding number
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()