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



from GastroModelValidator import GastroModelValidator
validate_or_test = GastroModelValidator.validate_or_test
calculate_metrics = GastroModelValidator.calculate_metrics
print_metrics = GastroModelValidator.print_metrics
training_curve = GastroModelValidator.training_curve
plot_confusion_matrix = GastroModelValidator.plot_confusion_matrix

# Set CuDNN benchmark for performance improvement on fixed input sizes
torch.backends.cudnn.benchmark = True

class Trainer:
    def __init__(self, train_root_dir, val_root_dir, test_root_dir, model_path, batch_size=32, max_epochs=150, lr=0.0001, n_classes=22, pin_memory=True,trans = None):
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
        # remember to use negative space programming to make the code more readable and maintainable
        if trans is None:
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
        #TODO validate the trans dictionary to ensure that it has the correct keys

        training_dataset = datasets.ImageFolder(train_root_dir, transform=trans["train"])
        validation_dataset = datasets.ImageFolder(val_root_dir, transform=trans["valid"])
        test_dataset = datasets.ImageFolder(test_root_dir, transform=trans["test"])

        #! pinning memory for faster data loading can also cause memory issues if the system has limited memory, for number of workers shouldn be a problem
        pin_memory = True if pin_memory else False
        self.training_loader = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory) 
        self.validation_loader = data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, pin_memory=pin_memory)
        self.test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=pin_memory)

        logging.info(f"Number of training images: {len(training_dataset)}")
        logging.info(f"Number of validation images: {len(validation_dataset)}")
        logging.info(f"Number of test images: {len(test_dataset)}")
        
        self.writer.add_text("Model", "densenet121: IMAGENET1K_V1 based model")


        if not os.path.exists(model_path):
            os.makedirs(model_path)

    def train_model(self, trial=None,max_epochs=150):
        if max_epochs is None:
            max_epochs = self.max_epochs
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

        for epoch in range(max_epochs):
            logging.info(f"Epoch {epoch + 1}/{max_epochs}")
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

            if True:  # Every 10 steps
                y_true = all_labels_d.cpu()
                y_pred = all_predictions_d.cpu()
                cm = confusion_matrix(y_true, y_pred)
                class_names = self.training_loader.dataset.classes
                cm_path = plot_confusion_matrix(cm, classes=class_names, save_path=self.model_path)
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
            val_loss, val_metrics, val_num_steps = validate_or_test(model, self.validation_loader,device=self.device, criterrion=criterion, validate=True, save_path=self.model_path)
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
            # Calculate and log the training speed
            training_speed = (time_elapsed % 60) / num_steps if time_elapsed > 0 else 0 # seconds per batch
            self.writer.add_scalar("Speed/train_per_batch", training_speed, epoch)
            self.writer.add_scalar("Speed/train_per_epoch", time_elapsed, epoch)

            if trial is not None:
                trial.report(val_metrics["micro_f1"], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        if trial is None:
            training_curve(epochs, lossesT, lossesV, save_path=self.model_path)
        return val_metrics["micro_f1"]


    def test_model(self):
        logging.info(f"Best model path: {self.best_model_path}")
        model = models.densenet121(weights=None)
        n_inputs = model.classifier.in_features
        model.classifier = nn.Linear(n_inputs, self.n_classes)
        model = model.to(self.device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        checkpoint = torch.load(self.best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        test_loss, test_metrics, test_num_steps = validate_or_test(model, self.test_loader, device=self.device, criterrion=criterion, validate=False, save_path=self.model_path)
        print_metrics(test_metrics, test_num_steps)

        with open(self.model_path + "test_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Test_loss"] + list(test_metrics.keys()))
            writer.writerow([test_loss] + list(test_metrics.values()))

        self.writer.add_scalar("Loss/test", test_loss, 0)
        self.writer.add_scalar("Micro_F1/test", test_metrics["micro_f1"], 0)
        self.writer.add_scalar("Macro_F1/test", test_metrics["macro_f1"], 0)
        
# extract features from the model
    def extract_features(self):
        model = models.densenet121(weights=None)
        n_inputs = model.classifier.in_features
        model.classifier = nn.Linear(n_inputs, self.n_classes)
        model = model.to(self.device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        checkpoint = torch.load(self.best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        model.eval()
        features = []
        labels = []
        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = model(images)
                features.append(outputs)
                labels.append(labels)

        features = torch.cat(features)
        labels = torch.cat(labels)
        return features, labels

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
    best_trainer.train_model(max_epochs=5)
    best_trainer.test_model()
