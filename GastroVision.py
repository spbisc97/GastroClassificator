import torch
import time
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import logging
import csv
from collections import defaultdict
import numpy as np
from torchvision import datasets, transforms, models
import sklearn.metrics as mtc
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools
import optuna
from torch.cuda.amp import GradScaler, autocast

# Set CuDNN benchmark for performance improvement on fixed input sizes
torch.backends.cudnn.benchmark = True

def calculate_metrics(y_true, y_pred):
    metrics = {
        "micro_precision": mtc.precision_score(y_true, y_pred, average="micro"),
        "micro_recall": mtc.recall_score(y_true, y_pred, average="micro"),
        "micro_f1": mtc.f1_score(y_true, y_pred, average="micro"),
        "macro_precision": mtc.precision_score(y_true, y_pred, average="macro"),
        "macro_recall": mtc.recall_score(y_true, y_pred, average="macro"),
        "macro_f1": mtc.f1_score(y_true, y_pred, average="macro"),
        "mcc": mtc.matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def print_metrics(metrics, num_steps):
    outputs = []
    for k, v in metrics.items():
        if k in ["dice_coeff", "dice", "bce"]:
            outputs.append(f"{k}:{v / num_steps:.4f}")
        else:
            outputs.append(f"{k}:{v:.2f}")
    logging.info(", ".join(outputs))


def training_curve(epochs, lossesT, lossesV):
    plt.plot(epochs, lossesT, "c", label="Training Loss")
    plt.plot(epochs, lossesV, "m", label="Validation Loss")
    plt.title("Training Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("train_val_epoch_curve.png")
    plt.close()


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues, plt_size=(10, 10)):
    plt.rcParams["figure.figsize"] = plt_size
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        logging.info("Normalized confusion matrix")
    else:
        logging.info("Confusion matrix, without normalization")
    
    logging.info(cm)
    
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Save the confusion matrix to a temporary file and return the path
    temp_path = "confusion_matrix.png"
    plt.savefig(temp_path)
    plt.close()
    return temp_path


def validate_or_test(model, data_loader, device, criterion, phase="validation"):
    model.eval()
    metrics = defaultdict(float)
    num_steps = 0
    total_loss = 0
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            num_steps += images.size(0)

            _, predicted = torch.max(outputs, 1)
            all_labels_d = torch.cat((all_labels_d, labels))
            all_predictions_d = torch.cat((all_predictions_d, predicted))

    y_true = all_labels_d.cpu()
    y_pred = all_predictions_d.cpu()

    metrics.update(calculate_metrics(y_true, y_pred))

    if phase == "test":
        cm = confusion_matrix(y_true, y_pred)
        class_names = data_loader.dataset.classes
        plot_confusion_matrix(cm, classes=class_names)
        logging.info(classification_report(y_true, y_pred, target_names=class_names))
        logging.info(f"Accuracy on the test set: {100 * (y_true == y_pred).sum().item() / num_steps:.2f}%")

    return total_loss / num_steps, metrics, num_steps


class Trainer:
    def __init__(self, train_root_dir, val_root_dir, test_root_dir, model_path, batch_size=32, max_epochs=150, lr=0.0001, n_classes=22):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.n_classes = n_classes
        self.model_path = model_path
        self.best_model_path = None
        self.val_f1_max = 0.0
        self.writer = SummaryWriter()  # TensorBoard writer
        self.scaler = GradScaler()  # For mixed precision training

        trans = {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33)),  # Added stretch
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "valid": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        training_dataset = datasets.ImageFolder(train_root_dir, transform=trans["train"])
        validation_dataset = datasets.ImageFolder(val_root_dir, transform=trans["valid"])
        test_dataset = datasets.ImageFolder(test_root_dir, transform=trans["test"])

        self.training_loader = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.validation_loader = data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        self.test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        logging.info(f"Number of training images: {len(training_dataset)}")
        logging.info(f"Number of validation images: {len(validation_dataset)}")
        logging.info(f"Number of test images: {len(test_dataset)}")

        if not os.path.exists(model_path):
            os.makedirs(model_path)

    def train_model(self, trial=None):
        lr = self.lr if trial is None else trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        weight_decay = 1e-4 if trial is None else trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        model = models.densenet121(weights='IMAGENET1K_V1')
        # Fine-tune the last layer for the specific number of classes
        n_inputs = model.classifier.in_features
        model.classifier = nn.Linear(n_inputs, self.n_classes)
        model = model.to(self.device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # DataParallel for using multiple GPUs

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, verbose=True)
        criterion = nn.CrossEntropyLoss()

        epochs = []
        lossesT = []
        lossesV = []

        for epoch in range(self.max_epochs):
            logging.info(f"Epoch {epoch + 1}/{self.max_epochs}")
            logging.info("-" * 10)

            since = time.time()
            model.train()
            train_loss = 0
            num_steps = 0
            all_labels_d = torch.tensor([], dtype=torch.long).to(self.device)
            all_predictions_d = torch.tensor([], dtype=torch.long).to(self.device)

            for step, (images, labels) in enumerate(self.training_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                train_loss += loss.item() * images.size(0)
                num_steps += images.size(0)
                _, predicted = torch.max(outputs, 1)
                all_labels_d = torch.cat((all_labels_d, labels))
                all_predictions_d = torch.cat((all_predictions_d, predicted))

                if step % 10 == 0:  # Every 10 steps
                    y_true = all_labels_d.cpu()
                    y_pred = all_predictions_d.cpu()
                    cm = confusion_matrix(y_true, y_pred)
                    class_names = self.training_loader.dataset.classes
                    cm_path = plot_confusion_matrix(cm, classes=class_names)

                    # Read the image and convert it to a format suitable for TensorBoard
                    image = plt.imread(cm_path)
                    self.writer.add_image("Confusion Matrix", image.transpose(2, 0, 1), epoch * len(self.training_loader) + step)

            y_true = all_labels_d.cpu()
            y_pred = all_predictions_d.cpu()

            train_metrics = calculate_metrics(y_true, y_pred)
            train_metrics["loss"] = train_loss / num_steps
            print_metrics(train_metrics, num_steps)

            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Micro_F1/train", train_metrics["micro_f1"], epoch)
            self.writer.add_scalar("Macro_F1/train", train_metrics["macro_f1"], epoch)

            model.eval()
            val_loss, val_metrics, val_num_steps = validate_or_test(model, self.validation_loader, self.device, criterion, phase="validation")
            scheduler.step(val_loss)

            self.writer.add_scalar("Loss/validation", val_loss, epoch)
            self.writer.add_scalar("Micro_F1/validation", val_metrics["micro_f1"], epoch)
            self.writer.add_scalar("Macro_F1/validation", val_metrics["macro_f1"], epoch)

            epochs.append(epoch)
            lossesT.append(train_loss / num_steps)
            lossesV.append(val_loss)

            logging.info(f"Validation loss: {val_loss:.3f}")
            print_metrics(val_metrics, val_num_steps)

            if trial is None:
                with open(self.model_path + "training_results.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    if epoch == 0:
                        writer.writerow(["Epoch"] + list(train_metrics.keys()) + ["Val_loss"] + list(val_metrics.keys()))
                    writer.writerow([epoch + 1] + list(train_metrics.values()) + [val_loss] + list(val_metrics.values()))

            if val_metrics["micro_f1"] >= self.val_f1_max:
                logging.info(f"Validation micro F1 increased ({self.val_f1_max:.6f} --> {val_metrics['micro_f1']:.6f}). Saving model...")
                self.best_model_path = os.path.join(self.model_path, f"best_model_epoch_{epoch + 1}.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": val_loss
                }, self.best_model_path)
                self.val_f1_max = val_metrics["micro_f1"]

            logging.info("-" * 10)
            time_elapsed = time.time() - since
            logging.info(f"Epoch time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

            if trial is not None:
                trial.report(val_metrics["micro_f1"], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        if trial is None:
            training_curve(epochs, lossesT, lossesV)
        return val_metrics["micro_f1"]


    def test_model(self):
        logging.info(f"Best model path: {self.best_model_path}")
        model = models.densenet121(pretrained=False)
        n_inputs = model.classifier.in_features
        model.classifier = nn.Linear(n_inputs, self.n_classes)
        model = model.to(self.device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        checkpoint = torch.load(self.best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        test_loss, test_metrics, test_num_steps = validate_or_test(model, self.test_loader, self.device, criterion, phase="test")
        print_metrics(test_metrics, test_num_steps)

        with open(self.model_path + "test_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Test_loss"] + list(test_metrics.keys()))
            writer.writerow([test_loss] + list(test_metrics.values()))

        self.writer.add_scalar("Loss/test", test_loss, 0)
        self.writer.add_scalar("Micro_F1/test", test_metrics["micro_f1"], 0)
        self.writer.add_scalar("Macro_F1/test", test_metrics["macro_f1"], 0)


def objective(trial):
    trainer = Trainer(train_root_dir="./DATASET/TRAIN", val_root_dir="./DATASET/VAL", test_root_dir="./DATASET/TEST", model_path="./models/")
    return trainer.train_model(trial)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=50)

    # logging.info(f"Best trial: {study.best_trial.value}")
    # logging.info(f"Best hyperparameters: {study.best_trial.params}")

    base_model_path = "./models/"
    date= time.strftime("%Y%m%d-%H%M%S")
    
    model_path= base_model_path + date + "/"

    best_trainer = Trainer(train_root_dir="./DATASET/TRAIN", val_root_dir="./DATASET/VAL", test_root_dir="./DATASET/TEST", model_path=model_path)
    # best_trainer.lr = study.best_trial.params['lr']
    best_trainer.train_model()
    best_trainer.test_model()
