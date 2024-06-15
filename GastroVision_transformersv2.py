import torch
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import logging
import csv
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import optuna
import numpy as np

from GastroModelValidator import GastroModelValidator

calculate_metrics = GastroModelValidator.calculate_metrics
print_metrics = GastroModelValidator.print_metrics
training_curve = GastroModelValidator.training_curve
plot_confusion_matrix = GastroModelValidator.plot_confusion_matrix
validate_or_test = GastroModelValidator.validate_or_test

torch.backends.cudnn.benchmark = True

class CustomDINOv2(nn.Module):
    def __init__(self, num_labels):
        super(CustomDINOv2, self).__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)  # Load pretrained DINOv2 model
        self.classifier = nn.Sequential(
            nn.Linear(384, 3072),
            nn.ReLU(),
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        features = self.dino(x)
        logits = self.classifier(features)
        return logits, features

    def get_attention_maps(self, x):
        attentions = self.dino.get_last_selfattention(x)
        return attentions

class Trainer:
    def __init__(self, train_root_dir, val_root_dir, test_root_dir, model_path, batch_size=16, max_epochs=150, lr=0.0001, n_classes=22, best_model_path=None, pin_memory=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.n_classes = n_classes
        self.model_path = model_path
        self.best_model_path = best_model_path

        self.val_f1_max = 0.0
        self.writer = SummaryWriter()  # TensorBoard writer

        self.trans = {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33)),  # Added stretch
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
            "valid": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
            "test": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        }

        training_dataset = datasets.ImageFolder(train_root_dir, transform=self.trans["train"])
        validation_dataset = datasets.ImageFolder(val_root_dir, transform=self.trans["valid"])
        test_dataset = datasets.ImageFolder(test_root_dir, transform=self.trans["test"])

        self.training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
        self.validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, pin_memory=pin_memory)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=pin_memory)

        logging.info(f"Number of training images: {len(training_dataset)}")
        logging.info(f"Number of validation images: {len(validation_dataset)}")
        logging.info(f"Number of test images: {len(test_dataset)}")

        self.writer.add_text("Model", "CustomDINOv2")

        if not os.path.exists(model_path):
            os.makedirs(model_path)

    def train_model(self, trial=None, max_epochs=None, continue_best=True):
        if max_epochs is None:
            max_epochs = self.max_epochs
        lr = self.lr if trial is None else trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        weight_decay = 1e-4 if trial is None else trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        if continue_best and self.best_model_path is not None:
            model = CustomDINOv2(num_labels=self.n_classes).to(self.device)
            checkpoint = torch.load(self.best_model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
        else:
            model = CustomDINOv2(num_labels=self.n_classes).to(self.device)
            
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

                logits, features = model(images)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                num_steps += images.size(0)
                _, predicted = torch.max(logits, 1)
                all_labels_d = torch.cat((all_labels_d, labels))
                all_predictions_d = torch.cat((all_predictions_d, predicted))

            y_true = all_labels_d.cpu().numpy()
            y_pred = all_predictions_d.cpu().numpy()

            cm = confusion_matrix(y_true, y_pred)
            class_names = self.training_loader.dataset.classes
            cm_path = plot_confusion_matrix(cm=cm, classes=class_names, save_path=self.model_path)
            # Read the image and convert it to a format suitable for TensorBoard
            image = plt.imread(cm_path)
            self.writer.add_image("Confusion Matrix", image.transpose(2, 0, 1), epoch * len(self.training_loader) + step)

            train_metrics = calculate_metrics(y_true, y_pred)
            train_metrics["loss"] = train_loss / num_steps
            print_metrics(train_metrics, num_steps)

            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Micro_F1/train", train_metrics["micro_f1"], epoch)
            self.writer.add_scalar("Macro_F1/train", train_metrics["macro_f1"], epoch)

            model.eval()
            val_loss, val_metrics, val_num_steps = validate_or_test(model, self.validation_loader, device=self.device, criterion=criterion, validate=True)
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
            training_speed = num_steps / time_elapsed if time_elapsed > 0 else 0  # images per second
            self.writer.add_scalar("Speed/train_images_per_sec", training_speed, epoch)
            self.writer.add_scalar("Speed/train_time_per_epoch", time_elapsed, epoch)

            if trial is not None:
                trial.report(val_metrics["micro_f1"], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        if trial is None:
            training_curve(epochs, lossesT, lossesV, save_path=self.model_path)
        return val_metrics["micro_f1"]

    def test_model(self):
        logging.info(f"Best model path: {self.best_model_path}")
        model = CustomDINOv2(num_labels=self.n_classes).to(self.device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        checkpoint = torch.load(self.best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        test_loss, test_metrics, test_num_steps = validate_or_test(model, self.test_loader, device=self.device, criterion=criterion, validate=False, save_path=self.model_path)
        print_metrics(test_metrics, test_num_steps)

        with open(self.model_path + "test_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Test_loss"] + list(test_metrics.keys()))
            writer.writerow([test_loss] + list(test_metrics.values()))

        self.writer.add_scalar("Loss/test", test_loss, 0)
        self.writer.add_scalar("Micro_F1/test", test_metrics["micro_f1"], 0)
        self.writer.add_scalar("Macro_F1/test", test_metrics["macro_f1"], 0)

    def extract_features_attentionmaps(self, image_paths):
        model = CustomDINOv2(num_labels=self.n_classes).to(self.device)
        model.eval()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        if self.best_model_path is not None:
            checkpoint = torch.load(self.best_model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)

        processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        features = []
        attentions = []

        with torch.no_grad():
            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(image).unsqueeze(0).to(self.device)
                logits, features = model(inputs)

                # Extract hidden states
                if features is not None:
                    features = features.cpu().numpy()
                    features.append(features)
                else:
                    logging.warning(f"No hidden states available for {image_path}")
                    features.append(None)

                # Extract attention maps
                attentions.append(model.get_attention_maps(inputs))

        return features, attentions

    def visualize_features(self, image_paths):
        features, attentions = self.extract_features_attentionmaps(image_paths)

        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            plt.imshow(image)
            plt.title(f"Image {i + 1}")
            plt.show()

            # Visualize hidden states
            if features[i] is not None:
                for j, feature_map in enumerate(features[i]):
                    if len(feature_map.shape) == 1:
                        feature_map = feature_map.reshape(24, 32)  # Reshape the 1D vector to a 2D grid for visualization
                    plt.imshow(feature_map, cmap="viridis")
                    plt.title(f"Feature map {j + 1}")
                    plt.show()
            else:
                logging.warning(f"No hidden states available for visualization for {image_path}")

            # Visualize attention maps
            if attentions[i] is not None:
                for j, attention_map in enumerate(attentions[i]):
                    attention_map = attention_map.squeeze(0).cpu().numpy()
                    attention_map = np.mean(attention_map, axis=0)  # Average over the heads
                    attention_map = attention_map.reshape(14, 14)  # Assuming the attention map is 14x14

                    # Resize the attention map to the size of the image
                    attention_map = np.kron(attention_map, np.ones((16, 16)))
                    plt.imshow(attention_map, cmap='viridis', alpha=0.6)
                    plt.title(f"Attention map {j + 1}")
                    plt.show()

        return features, attentions

    # Return the best model already trained as a model object
    def get_best_model(self):
        model = CustomDINOv2(num_labels=self.n_classes).to(self.device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        checkpoint = torch.load(self.best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def predict(self, image_path, return_class=False):
        model = self.get_best_model()
        processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image).unsqueeze(0).to(self.device)
        logits, _ = model(inputs)
        if return_class:
            index = torch.argmax(logits).item()
            return self.test_loader.dataset.classes[index]
        else:
            return logits

    def extract_features(self, image_path):
        model = self.get_best_model()
        processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image).unsqueeze(0).to(self.device)
        _, features = model(inputs)
        features = features.cpu().numpy()
        return features


def objective(trial):
    trainer = Trainer(train_root_dir="./DATASET/TRAIN", val_root_dir="./DATASET/VAL", test_root_dir="./DATASET/TEST", model_path="./models/")
    return trainer.train_model(trial, max_epochs=8)  # Shorter training for hyperparameter tuning


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=2)

    # logging.info(f"Best trial: {study.best_trial.value}")
    # logging.info(f"Best hyperparameters: {study.best_trial.params}")

    base_model_path = "./models/"
    date = time.strftime("%Y%m%d-%H%M%S")
    model_path = base_model_path + date + "/"
    

    best_trainer = Trainer(train_root_dir="./DATASET/TRAIN", val_root_dir="./DATASET/VAL", test_root_dir="./DATASET/TEST", model_path=model_path,batch_size=32,best_model_path='models/20240614-232758/best_model_epoch_13.pth')
    # best_trainer.lr = study.best_trial.params['lr']
    best_trainer.train_model(max_epochs=15,continue_best=False)  # Full training with the best hyperparameters
    best_trainer.test_model()


    # Keep it open in the terminal and interact with python
    import code
    code.interact(local=locals())
