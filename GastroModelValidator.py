# import all the necessary libraries
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import  autocast
from collections import defaultdict
from sklearn import metrics as mtc
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import os
import torchvision


# Class to validate the model

class GastroModelValidator:
    def __init__(self):
        pass
    
    @staticmethod
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

    @staticmethod
    def print_metrics(metrics, num_steps):
        outputs = []
        for k, v in metrics.items():
            if k in ["dice_coeff", "dice", "bce"]:
                outputs.append(f"{k}:{v / num_steps:.4f}")
            else:
                outputs.append(f"{k}:{v:.2f}")
        logging.info(", ".join(outputs))
    
    @staticmethod
    def training_curve(epochs, lossesT, lossesV, save_path='.'):
        plt.plot(epochs, lossesT, "c", label="Training Loss")
        plt.plot(epochs, lossesV, "m", label="Validation Loss")
        plt.title("Training Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        save_path = os.path.join(save_path,"train_val_epoch_curve.png")
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrix( cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues, plt_size=(10, 10), save_path='.'):
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
        save_path = os.path.join(save_path,"confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        return save_path



    @staticmethod
    def validate_or_test(model,dataloader,  validate=True, device="cuda",criterion=None,save_path='.',VIT_model=None): #ViT model : vit-base-patch16-224,dino_vit
        #use negative space programming to check the inputs
        ViTModels = ['vit-base-patch16-224','dino_vit','vittorchvision','ViTForImageClassificationCustom','ViTForImageClassification']
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        if device == "cuda":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cpu":
                logging.warning("CUDA is not available, falling back to CPU")
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()
        metrics = defaultdict(float)
        num_steps = 0
        total_loss = 0
        all_labels_d = torch.tensor([], dtype=torch.long).to(device)
        all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                with autocast():
                    #if is an instance of transformers learning model
                    if VIT_model is not None:
                        if VIT_model in ViTModels:
                            if VIT_model == 'vit-base-patch16-224':
                                outputs,_ = model(images)
                            if VIT_model == 'vittorchvision':
                                outputs,_ = model(images)
                            elif VIT_model == 'dino_vit':
                                outputs = model(images)
                            elif VIT_model == 'ViTForImageClassificationCustom':
                                outputs = model(images).logits
                            elif VIT_model == 'ViTForImageClassification':
                                outputs = model(images).logits
                        else:
                            logging.error("Please provide a valid ViT model")
                            return
                    else:
                        outputs = model(images)
                    loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                num_steps += images.size(0)
                _, predicted = torch.max(outputs, 1)
                all_labels_d = torch.cat((all_labels_d, labels))
                all_predictions_d = torch.cat((all_predictions_d, predicted))
        y_true = all_labels_d.cpu()
        y_pred = all_predictions_d.cpu()
        metrics.update(GastroModelValidator.calculate_metrics(y_true, y_pred))
        if not validate:
            cm = confusion_matrix(y_true, y_pred)
            class_names = dataloader.dataset.classes
            GastroModelValidator.plot_confusion_matrix(cm, classes=class_names,save_path=save_path)
            logging.info(classification_report(y_true, y_pred, target_names=class_names,zero_division=0))
        logging.info(f"Accuracy on the {'validation' if validate else 'test'} set: {100 * (y_true == y_pred).sum().item() / num_steps:.2f}%")
        return total_loss / num_steps, metrics, num_steps