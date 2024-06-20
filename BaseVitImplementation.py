# %%
"""
<a target="_blank" href="https://colab.research.google.com/github/spbisc97/GastroClassificator/blob/main/BaseVitImplementation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from torchvision import transforms, datasets
import logging
from torch import nn # if you want to use the nn module for custom classification head
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn import metrics as mtc
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, ViTConfig
# if you want to use custom classification head
from transformers.modeling_outputs import CausalLMOutput,ImageClassifierOutput
from torch.cuda.amp import GradScaler, autocast
# from tqdm import tqdm
# if using the notebook
def is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm



# from tqdm.notebook import tqdm

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


# %%

# Load the feature extractor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')


# Function to load datasets
def load_dataset(train_root_dir, val_root_dir, test_root_dir, transform=None,normalize=True, feature_extractor=feature_extractor):
    #if transform is None use the VitImageProcessor
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if transform is None:
        transform = default_transform
    if normalize:
        transform = transforms.Compose([
            transform,
            transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ])
        default_transform = transforms.Compose([
            default_transform,
            transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ])
    train_dataset = datasets.ImageFolder(train_root_dir, transform=transform)
    logger.info("Training dataset loaded with %d samples", len(train_dataset))
    val_dataset = datasets.ImageFolder(val_root_dir, transform=default_transform)
    logger.info("Validation dataset loaded with %d samples", len(val_dataset))
    test_dataset = datasets.ImageFolder(test_root_dir, transform=default_transform)
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

# %%

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(10, 10), font_size=12,save_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'figure.figsize': figsize})
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    


# %%

# Training arguments
metric_name = "micro_f1"
args = TrainingArguments(
    "test-GastroVision",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=1e-4,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='./logs',
    logging_steps=30,  # Log every 10 steps
    fp16=True,  # Enable mixed precision training
)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33)),  # Added stretch
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Load datasets
train_dataset, val_dataset, test_dataset = load_dataset(train_root_dir, val_root_dir, test_root_dir,transform=transform,normalize=False)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

# %%
class CustomViTForImageClassification(ViTForImageClassification):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        # ... # add more layers if needed
        # ...
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        # ... # add more layers if needed
        # ...
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        loss = nn.CrossEntropyLoss()(logits, labels)
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
    
    
    
    

# %%
# Load the model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(get_labels(train_dataset)),attn_implementation='eager')
#model = CustomViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(get_labels(train_dataset)),attn_implementation='eager')
# Move the model to the appropriate device
model.to(device)

# %%
# Define optimizer and learning rate scheduler for custom training loop
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4) #add verbose=True to see the learning rate change

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

# Custom training loop
def custom_train(trainer, model, train_loader,val_loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    global_step = 0
    for epoch in range(trainer.args.num_train_epochs):
        for step, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            with autocast():
                outputs = model(pixel_values=images, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            global_step += 1

            if global_step % trainer.args.logging_steps == 0:
                avg_loss = total_loss / global_step
                tqdm.write(f"Step {global_step}, Loss: {avg_loss:.4f}")
                # logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}") # Uncomment this line if you want to log the loss to the logger

        scheduler.step()
        avg_loss = total_loss / (epoch + 1)
        logger.info(f"Epoch {epoch+1}/{trainer.args.num_train_epochs}, Train Loss: {avg_loss:.4f}")
        
        # Evaluate the model
        eval_results = trainer.evaluate(eval_dataset=val_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{trainer.args.num_train_epochs}, Eval {eval_results}")
        
        


# Initialize the trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    data_collator=custom_collate_fn,
    # data_collator=lambda x: {"pixel_values": torch.stack([item[0] for item in x]), "labels": torch.tensor([item[1] for item in x])},
    compute_metrics=compute_metrics,
)



# %%
# Train the model
# Train the model using the custom training loop
custom_train(trainer, model, train_loader, val_loader,optimizer, scheduler, scaler, device)
# trainer.train()

# %%


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


# %%

plot_confusion_matrix(conf_matrix, get_labels(test_dataset), normalize=False, title="Normalized Confusion Matrix", figsize=(15, 15),font_size=12)

plt.show()
# %%

#convert to notebook
# ipynb-py-convert BaseVitImplementation.py BaseVitImplementation.ipynb
# or the other way around
# ipynb-py-convert BaseVitImplementation.ipynb BaseVitImplementation.py