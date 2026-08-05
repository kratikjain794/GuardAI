class EarlyStopping:
    """
    Early Stopping based on validation metric.

    Example:
        early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.001
        )
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
    ):

        self.patience = patience

        self.min_delta = min_delta

        self.best_score = None

        self.counter = 0

        self.should_stop = False

    # ======================================
    # Step
    # ======================================

    def step(
        self,
        current_score,
    ):

        # First epoch
        if self.best_score is None:

            self.best_score = current_score

            return False

        # Improvement
        if (
            current_score
            >
            self.best_score
            + self.min_delta
        ):

            self.best_score = current_score

            self.counter = 0

            return False

        # No improvement
        self.counter += 1

        print(
            f"No improvement "
            f"({self.counter}/{self.patience})"
        )

        if self.counter >= self.patience:

            self.should_stop = True

            print("\n================================")

            print("Early Stopping Triggered")

            print("================================")

            return True

        return False

    # ======================================
    # Reset
    # ======================================

    def reset(self):

        self.best_score = None

        self.counter = 0

        self.should_stop = False



if __name__ == "__main__":

    early_stop = EarlyStopping(
        patience=3,
    )

    validation_scores = [

        0.50,

        0.55,

        0.58,

        0.58,

        0.57,

        0.56,

        0.56,

    ]

    for epoch, score in enumerate(
        validation_scores,
        start=1,
    ):

        print(
            f"\nEpoch {epoch}"
        )

        print(
            f"Validation F1 : {score}"
        )

        if early_stop.step(score):

            print(
                f"\nTraining stopped at epoch {epoch}"
            )

            break