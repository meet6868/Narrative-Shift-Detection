from __future__ import annotations

import matplotlib.pyplot as plt


def plot_training_loss(history, save_path, title="TCL Training Loss"):
    """Plot shared training curves for AP1/AP2/AP4/AP5 histories."""
    if not history:
        return

    epochs = history.get("epoch")
    if not epochs:
        epochs = list(range(1, len(history.get("loss", [])) + 1))

    losses = history.get("loss", [])
    if not losses:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, losses, marker="o", linewidth=1.8, label="total")

    if history.get("temporal_loss"):
        plt.plot(epochs, history["temporal_loss"], linewidth=1.2, label="temporal")
    if history.get("topic_sep_loss"):
        plt.plot(epochs, history["topic_sep_loss"], linewidth=1.2, label="topic_sep")
    if history.get("hard_neg_loss"):
        plt.plot(epochs, history["hard_neg_loss"], linewidth=1.2, label="hard_neg")
    if history.get("entity_loss"):
        plt.plot(epochs, history["entity_loss"], linewidth=1.2, label="entity")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)

    has_components = any(
        history.get(key) for key in ["temporal_loss", "topic_sep_loss", "hard_neg_loss", "entity_loss"]
    )
    if has_components:
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_ap5_training_dashboard(history, save_path, show=True):
    """AP5-style dashboard with total/component/LR/final component bars."""
    if not history or not history.get("loss"):
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["loss"])
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history.get("temporal_loss", []), label="Temporal")
    axes[0, 1].plot(history.get("topic_sep_loss", []), label="Topic Sep")
    axes[0, 1].plot(history.get("hard_neg_loss", []), label="Hard Neg")
    if history.get("entity_loss"):
        axes[0, 1].plot(history["entity_loss"], label="Entity")
    axes[0, 1].set_title("Component Losses")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history.get("lr", []))
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LR")
    axes[1, 0].grid(True, alpha=0.3)

    component_labels = ["Temporal", "Topic Sep", "Hard Neg"]
    component_values = [
        history.get("temporal_loss", [0.0])[-1],
        history.get("topic_sep_loss", [0.0])[-1],
        history.get("hard_neg_loss", [0.0])[-1],
    ]
    if history.get("entity_loss"):
        component_labels.append("Entity")
        component_values.append(history["entity_loss"][-1])

    axes[1, 1].bar(component_labels, component_values)
    axes[1, 1].set_title("Final Component Losses")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)