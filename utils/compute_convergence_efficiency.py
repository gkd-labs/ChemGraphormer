def compute_convergence_training_efficiency(training_time, validation_mae):
    """
    Computes convergence efficiency as:
        (training_time_at_patience_or_best_val * best_validation_MAE)

    Args:
        training_time (float): Total minutes until patience trigger.
        validation_mae (float): Best validation MAE achieved.

    Returns:
        float: Convergence efficiency score (lower is better).
    """
    if validation_mae <= 0:
        raise ValueError("validation_mae must be > 0")

    return (training_time * validation_mae)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Compute convergence efficiency (total training time x best Val MAE) from an epoch log CSV."
    )
    parser.add_argument(
        "--log_path", type=str, required=True,
        help="Path to the epoch log CSV produced by a training pipeline (must have 'Epoch Time (min)' and 'Val MAE' columns)."
    )
    args = parser.parse_args()

    logs = pd.read_csv(args.log_path)
    total_time = logs["Epoch Time (min)"].sum()
    best_mae = logs["Val MAE"].min()

    efficiency = compute_convergence_training_efficiency(total_time, best_mae)

    print(f"Total training time (min): {total_time:.2f}")
    print(f"Best Val MAE:              {best_mae:.5f}")
    print(f"Convergence Efficiency:    {efficiency:.5f}")