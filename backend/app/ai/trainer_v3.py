import os

import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from torch.optim import Adam

from torch.optim.lr_scheduler import ReduceLROnPlateau

from app.ai.metrics import (
    compute_metrics,
)

from app.ai.metrics import (
    compute_confusion_matrix,
    compute_report,
)

from app.ai.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    checkpoint_exists,
)

from app.ai.early_stopping import (
    EarlyStopping,
)


class TrainerV3:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        learning_rate=1e-3,
        epochs=40,
        model_dir="trained_models",
        best_model_name="best_model.pth",
        last_model_name="last_model.pth",
        log_dir="runs/default",
        training_name="Training",
        criterion=None,
    ):
        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        self.model = model.to(

            self.device

        )

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.test_loader = test_loader

        self.epochs = epochs

        self.model_dir = model_dir

        os.makedirs(

            model_dir,

            exist_ok=True,

        )

        self.best_model_path = os.path.join(
            model_dir,
            best_model_name,
        )

        self.last_model_path = os.path.join(
            model_dir,
            last_model_name,
        )

        if criterion is None:

            self.criterion = nn.CrossEntropyLoss()

        else:

            self.criterion = criterion

        self.optimizer = Adam(

            self.model.parameters(),

            lr=learning_rate,

        )

        self.scheduler = ReduceLROnPlateau(

            self.optimizer,

            mode="max",

            factor=0.5,

            patience=3,

        )

        self.early_stopping = EarlyStopping(
            patience=10,
        )

        self.writer = SummaryWriter(
            log_dir=log_dir
        )

        self.start_epoch = 1

        self.best_f1 = 0.0

        self.training_name = training_name

        if checkpoint_exists(

            self.last_model_path

        ):

            print(

                "\nResuming Training...\n"

            )

            epoch, best = load_checkpoint(

                self.last_model_path,

                self.model,

                self.optimizer,

                self.device,

            )

            self.start_epoch = epoch + 1

            self.best_f1 = best

        print("\n============================")

        print("GuardIA Trainer V3")

        print("============================")

        print(

            "Device :",

            self.device,

        )

        print(

            "Epochs :",

            self.epochs,

        )

        print(

            "Starting Epoch :",

            self.start_epoch,

        )

        print(

            "Best F1 :",

            self.best_f1,

        )


    # ==========================================
    # Train One Epoch
    # ==========================================

    def train_epoch(
        self,
        epoch,
    ):

        self.model.train()

        running_loss = 0.0

        predictions = []

        labels = []

        progress = tqdm(

            self.train_loader,

            desc=f"Epoch {epoch}/{self.epochs}",

            leave=False,

        )

        for features, target in progress:

            features = features.to(

                self.device

            )

            target = target.to(

                self.device

            )

            # ------------------------------
            # Forward
            # ------------------------------

            outputs = self.model(

                features

            )

            loss = self.criterion(

                outputs,

                target,

            )

            # ------------------------------
            # Backward
            # ------------------------------

            self.optimizer.zero_grad()

            loss.backward()

            # ------------------------------
            # Gradient Clipping
            # ------------------------------

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                max_norm=1.0,

            )

            self.optimizer.step()

            # ------------------------------
            # Statistics
            # ------------------------------

            running_loss += loss.item()

            predicted = torch.argmax(

                outputs,

                dim=1,

            )

            predictions.extend(

                predicted.detach()

                .cpu()

                .numpy()

            )

            labels.extend(

                target.detach()

                .cpu()

                .numpy()

            )

            progress.set_postfix(

                loss=f"{loss.item():.4f}"

            )

        # --------------------------------------

        average_loss = (

            running_loss

            / len(self.train_loader)

        )

        metrics = compute_metrics(

            labels,

            predictions,

        )

        confusion = compute_confusion_matrix(
            labels,
            predictions,
        )

        print("\n================================")
        print("CONFUSION MATRIX")
        print("================================")
        print(confusion)

        print("\n================================")
        print("CLASSIFICATION REPORT")
        print("================================")

        print(
        compute_report(
           labels,
           predictions,
           target_names=[
              "normal",
              "distress",
            ],
        )
    )

        # --------------------------------------

        self.writer.add_scalar(

            "Train/Loss",

            average_loss,

            epoch,

        )

        self.writer.add_scalar(

            "Train/Accuracy",

            metrics["accuracy"],

            epoch,

        )

        self.writer.add_scalar(

            "Train/F1",

            metrics["f1"],

            epoch,

        )

        print("\n============================")

        print(

            f"Epoch {epoch}"

        )

        print("============================")

        print(

            f"Train Loss : "

            f"{average_loss:.4f}"

        )

        print(

            f"Accuracy   : "

            f"{metrics['accuracy']*100:.2f}%"

        )

        print(

            f"Precision  : "

            f"{metrics['precision']:.4f}"

        )

        print(

            f"Recall     : "

            f"{metrics['recall']:.4f}"

        )

        print(

            f"F1 Score   : "

            f"{metrics['f1']:.4f}"

        )

        return (

            average_loss,

            metrics,

        )

    # ==========================================
    # Validation
    # ==========================================

    def validate(
        self,
        epoch,
    ):

        self.model.eval()

        running_loss = 0.0

        predictions = []

        labels = []

        with torch.no_grad():

            progress = tqdm(

                self.val_loader,

                desc="Validation",

                leave=False,

            )

            for features, target in progress:

                features = features.to(

                    self.device

                )

                target = target.to(

                    self.device

                )

                outputs = self.model(

                    features

                )

                loss = self.criterion(

                    outputs,

                    target,

                )

                running_loss += loss.item()

                

                predicted = torch.argmax(

                    outputs,

                    dim=1,

                )

                predictions.extend(

                    predicted.cpu().numpy()

                )

                labels.extend(

                    target.cpu().numpy()

                )

        average_loss = (

            running_loss

            / len(self.val_loader)

        )

        metrics = compute_metrics(

            labels,

            predictions,

        )


        self.writer.add_scalar(

            "Validation/Loss",

            average_loss,

            epoch,

        )

        self.writer.add_scalar(

            "Validation/F1",

            metrics["f1"],

            epoch,

        )

        print("\nValidation")

        print("----------------------------")

        print(

            f"Loss      : {average_loss:.4f}"

        )

        print(

            f"Accuracy  : {metrics['accuracy']*100:.2f}%"

        )

        print(

            f"Precision : {metrics['precision']:.4f}"

        )

        print(

            f"Recall    : {metrics['recall']:.4f}"

        )

        print(

            f"F1 Score  : {metrics['f1']:.4f}"

        )

        return (

            average_loss,

            metrics,

        )


    # ==========================================
    # Test
    # ==========================================

    def test(self):

        self.model.eval()

        predictions = []

        labels = []

        running_loss = 0.0

        with torch.no_grad():

            progress = tqdm(

                self.test_loader,

                desc="Testing",

                leave=False,

            )

            for features, target in progress:

                features = features.to(

                    self.device

                )

                target = target.to(

                    self.device

                )

                outputs = self.model(

                    features

                )

                loss = self.criterion(

                    outputs,

                    target,

                )

                running_loss += loss.item()

                predicted = torch.argmax(

                    outputs,

                    dim=1,

                )

                predictions.extend(

                    predicted.cpu().numpy()

                )

                labels.extend(

                    target.cpu().numpy()

                )

        average_loss = (

            running_loss

            / len(self.test_loader)

        )

        metrics = compute_metrics(

            labels,

            predictions,

        )


        confusion = compute_confusion_matrix(
            labels,
            predictions,
        )

        print("\n================================")
        print("FINAL TEST CONFUSION MATRIX")
        print("================================")
        print(confusion)

        print("\n================================")
        print("FINAL TEST CLASSIFICATION REPORT")
        print("================================")

        print(
            compute_report(
                labels,
                predictions,
                target_names=[
                    "normal",
                    "distress",
                ], 
            )
         )


        print("\n================================")

        print("FINAL TEST RESULTS")

        print("================================")

        print(

            f"Loss      : {average_loss:.4f}"

        )

        print(

            f"Accuracy  : {metrics['accuracy']*100:.2f}%"

        )

        print(

            f"Precision : {metrics['precision']:.4f}"

        )

        print(

            f"Recall    : {metrics['recall']:.4f}"

        )

        print(

            f"F1 Score  : {metrics['f1']:.4f}"

        )

        return metrics


    # ==========================================
    # Complete Training
    # ==========================================

    def fit(self):

        print("\n================================")
        print(f"Starting {self.training_name}")
        print("================================")

        for epoch in range(
            self.start_epoch,
            self.epochs + 1,
        ):

            # -----------------------------
            # Train
            # -----------------------------

            train_loss, train_metrics = (
                self.train_epoch(epoch)
            )

            # -----------------------------
            # Validation
            # -----------------------------

            val_loss, val_metrics = (
                self.validate(epoch)
            )

            # -----------------------------
            # Learning Rate Scheduler
            # -----------------------------

            self.scheduler.step(
                val_metrics["f1"]
            )

            # -----------------------------
            # Save Last Checkpoint
            # -----------------------------

            save_checkpoint(

                self.last_model_path,

                self.model,

                self.optimizer,

                epoch,

                self.best_f1,

            )

            # -----------------------------
            # Save Best Model
            # -----------------------------

            if (

                val_metrics["f1"]

                >

                self.best_f1

            ):

                self.best_f1 = (

                    val_metrics["f1"]

                )

                save_checkpoint(

                    self.best_model_path,

                    self.model,

                    self.optimizer,

                    epoch,

                    self.best_f1,

                )

                print("\nBest model saved.")

            # -----------------------------
            # Early Stopping
            # -----------------------------

            if self.early_stopping.step(

                val_metrics["f1"]

            ):

                break

        # =====================================
        # Load Best Model
        # =====================================

        print("\nLoading Best Model...")

        load_checkpoint(

            self.best_model_path,

            self.model,

            device=self.device,

        )

        # =====================================
        # Final Test
        # =====================================

        self.test()

        self.writer.close()

        print("\n================================")

        print(f"{self.training_name} Finished Successfully")

        print("================================")
        