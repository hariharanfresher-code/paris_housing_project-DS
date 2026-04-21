"""src/utils.py — shared helper functions for the Paris Housing DS project."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = "Blues_d"
ACCENT  = "#1a73e8"


def load_data(path: str = "../data/paris_housing_prices_dataset.csv") -> pd.DataFrame:
    """Load and return the raw dataset."""
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def plot_price_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of Price_EUR."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["Price_EUR"], bins=40, color=ACCENT, edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribution of Price (€)", fontsize=14)
    axes[0].set_xlabel("Price (€)")
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x/1e6:.1f}M"))

    axes[1].hist(np.log1p(df["Price_EUR"]), bins=40, color=ACCENT, edgecolor="white", alpha=0.85)
    axes[1].set_title("Distribution of log(Price + 1)", fontsize=14)
    axes[1].set_xlabel("log(Price + 1)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("../outputs/price_distribution.png", dpi=150)
    plt.show()


def evaluate_model(name: str, y_true, y_pred) -> dict:
    """Print and return regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{'─'*40}")
    print(f"  Model : {name}")
    print(f"  RMSE  : €{rmse:,.0f}")
    print(f"  MAE   : €{mae:,.0f}")
    print(f"  R²    : {r2:.4f}")
    print(f"{'─'*40}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def plot_feature_importance(model, feature_names: list, top_n: int = 10) -> None:
    """Bar chart of the top-N feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    importances.plot(kind="barh", ax=ax, color=ACCENT)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("../outputs/feature_importance.png", dpi=150)
    plt.show()


def plot_residuals(y_true, y_pred, model_name: str = "") -> None:
    """Residual scatter + histogram."""
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, color=ACCENT, s=18)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Residuals vs Predicted — {model_name}", fontsize=13)
    axes[0].set_xlabel("Predicted Price (€)")
    axes[0].set_ylabel("Residual (€)")

    axes[1].hist(residuals, bins=40, color=ACCENT, edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_title("Residual Distribution", fontsize=13)
    axes[1].set_xlabel("Residual (€)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"../outputs/residuals_{model_name.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()
