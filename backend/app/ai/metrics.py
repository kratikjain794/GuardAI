from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ==========================================
# Compute Metrics
# ==========================================

def compute_metrics(
    y_true,
    y_pred,
    average="macro",
):

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        ),
    }

    return metrics


# ==========================================
# Confusion Matrix
# ==========================================

def compute_confusion_matrix(
    y_true,
    y_pred,
):

    return confusion_matrix(
        y_true,
        y_pred,
    )


# ==========================================
# Classification Report
# ==========================================

def compute_report(
    y_true,
    y_pred,
    target_names=None,
):

    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )


# ==========================================
# Print Metrics
# ==========================================

def print_metrics(
    metrics,
):

    print("\n==============================")

    print("Evaluation Metrics")

    print("==============================")

    print(
        f"Accuracy : {metrics['accuracy']*100:.2f}%"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {metrics['f1']:.4f}"
    )


if __name__ == "__main__":

    y_true = [

        0,1,2,3,4,5,6

    ]

    y_pred = [

        0,1,2,2,4,5,6

    ]

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    print_metrics(
        metrics,
    )

    print("\nConfusion Matrix\n")

    print(

        compute_confusion_matrix(
            y_true,
            y_pred,
        )

    )

    print("\nClassification Report\n")

    print(

        compute_report(
            y_true,
            y_pred,
        )

    )
