import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import example_groups

def plot_forgetting_events(forgetting_results, total_samples):
    # Count freqs of forgetting events
    forgetting_counts = {-1: 0}
    for value in forgetting_results.values():
        if value not in forgetting_counts:
            forgetting_counts[value] = 0
        forgetting_counts[value] += 1

    # Separate not-learned from other labels
    forgetting_keys = sorted([k for k in forgetting_counts.keys() if k != -1])
    forgetting_labels = forgetting_keys + ["Not Learned"]

    # Calculate percentages
    forgetting_values = [forgetting_counts[k] / total_samples * 100 for k in forgetting_keys]
    forgetting_values.append(forgetting_counts[-1] / total_samples * 100)

    # Plot chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(forgetting_labels)), forgetting_values, color='gray')

    plt.xticks(range(len(forgetting_labels)), forgetting_labels)
    plt.xlabel("Forgetting Events")
    plt.ylabel("% of Samples")
    plt.title("Forgetting Events Bar Chart")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return plt

def plot_average_loss_over_epochs(epoch_losses):
    # Plot the average loss per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-', color='b')

    plt.title("Loss over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)

    return plt

def plot_data_map(data_map_stats):
    confidence = np.array(data_map_stats["confidence"])
    variability = np.array(data_map_stats["variability"])
    correctness = np.array(data_map_stats["correctness"])

    # --- First Figure: Scatter Plot ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    scatter = ax1.scatter(variability, confidence, c=correctness, cmap="coolwarm", alpha=0.6, edgecolors="k")
    ax1.set_xlabel("Variability")
    ax1.set_ylabel("Average Confidence")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Data Map: Variability vs. Confidence")
    fig1.colorbar(scatter, ax=ax1, label="Correctness")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Second Figure: Histograms ---
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Density Histogram: Confidence
    sns.histplot(confidence, bins=20, ax=axes[0], color="blue")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Density Distribution of Confidence")

    # Density Histogram: Correctness
    sns.histplot(correctness, bins=20, ax=axes[1], color="green")
    axes[1].set_xlabel("Correctness")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Density Distribution of Correctness")

    # Density Histogram: Variability
    sns.histplot(variability, bins=20, ax=axes[2], color="red")
    axes[2].set_xlabel("Variability")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Density Distribution of Variability")

    plt.tight_layout()

    return fig1, fig2

def plot_losses_and_accuracies(dataset, all_losses, true_labels, predictions):
    results = example_groups.get_example_group_results(dataset)
    all_losses = np.array(all_losses)
    all_losses = np.nan_to_num(all_losses, nan=0.0)
    
    # Track the accuracies for all epochs
    all_accuracies = []
    for epoch in range(len(predictions)):
        epoch_accuracy = (np.array(predictions[epoch]) == np.array(true_labels)).astype(float)
        all_accuracies.append(epoch_accuracy)
    all_accuracies = np.array(all_accuracies)
    all_accuracies = np.nan_to_num(all_accuracies, nan=0.0)

    line_styles = ['-', '--', '-.']
    colors = ['b', 'g', 'r']

    # Figure 1: Losses
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for idx, (group, indices) in enumerate(results.items()):
        if not indices:
            print(f"No examples in the group {group}.")
            continue

        group_losses = all_losses[:, indices]
        avg_loss_per_epoch = np.nanmean(group_losses, axis=1)

        ax1.plot(range(1, len(avg_loss_per_epoch) + 1), 
                 avg_loss_per_epoch,
                 label=f"{group} (Loss)",
                 linestyle=line_styles[idx % len(line_styles)],
                 color=colors[idx % len(colors)])

    overall_avg_loss_per_epoch = np.nanmean(all_losses, axis=1)
    ax1.plot(range(1, len(overall_avg_loss_per_epoch) + 1), 
             overall_avg_loss_per_epoch,
             label='Overall Average Loss',
             linestyle='-',
             color=colors[-1],
             linewidth=2.5)
    ax1.set_title("Loss Trajectories")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Figure 2: Accuracies
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for idx, (group, indices) in enumerate(results.items()):
        if not indices:
            continue
        group_accuracies = all_accuracies[:, indices]
        avg_accuracy_per_epoch = np.mean(group_accuracies, axis=1)

        ax2.plot(range(1, len(avg_accuracy_per_epoch) + 1), 
                 avg_accuracy_per_epoch,
                 label=f"{group} (Accuracy)",
                 linestyle=line_styles[idx % len(line_styles)],
                 color=colors[idx % len(colors)])

    overall_avg_accuracy_per_epoch = np.nanmean(all_accuracies, axis=1)
    ax2.plot(range(1, len(overall_avg_accuracy_per_epoch) + 1),
             overall_avg_accuracy_per_epoch,
             label='Overall Average Accuracy',
             linestyle='-',
             color=colors[-1],
             linewidth=2.5)
    ax2.set_title("Accuracy Trajectories")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    return fig1, fig2

def plot_aum(aum_scores, true_labels, predictions):
    """
    Plot AUM trajectories for correctly and incorrectly labeled examples using dictionaries.

    Args:
        aum_scores: Dictionary {sample_idx: list of AUM scores across epochs}
    """
    num_epochs = len(next(iter(aum_scores.values())))
    correct_aum = []
    mislabeled_aum = []

    for epoch in range(num_epochs):
        epoch_correct_scores = []
        epoch_mislabeled_scores = []

        for sample_idx in aum_scores.keys():
            aum_score = aum_scores[sample_idx][epoch]
            true_label = true_labels[sample_idx]
            pred_label = predictions[epoch][sample_idx]

            if pred_label is not None:
                if pred_label == true_label:
                    epoch_correct_scores.append(aum_score)
                else:
                    epoch_mislabeled_scores.append(aum_score)

        # Replace 0 with NaN to avoid skewed averages
        epoch_correct_scores = np.array(epoch_correct_scores)
        epoch_mislabeled_scores = np.array(epoch_mislabeled_scores)
        epoch_correct_scores[epoch_correct_scores == 0] = np.nan
        epoch_mislabeled_scores[epoch_mislabeled_scores == 0] = np.nan

        correct_aum.append(np.nanmean(epoch_correct_scores) if len(epoch_correct_scores) > 0 else np.nan)
        mislabeled_aum.append(np.nanmean(epoch_mislabeled_scores) if len(epoch_mislabeled_scores) > 0 else np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(correct_aum, label="Correctly Labeled", color="green")
    plt.plot(mislabeled_aum, label="Mislabeled", color="red", linestyle="dashed")
    plt.title("AUM Trajectories for Correctly Labeled vs Mislabeled Examples")
    plt.xlabel("Epoch")
    plt.ylabel("AUM Score")
    plt.legend()
    plt.grid(True)
    return plt