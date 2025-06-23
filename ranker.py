import numpy as np

class ScoreRanker:
    """
    Analyses raw training dynamics scores to sort data samples from easiest to hardest.

     The output for each metric is a list of arrays, where each
    array contains the dataset indices sorted from easiest to hardest for a given epoch.
    """
    def __init__(self, stats: dict):
        """
        Initialises the ranker and processes the provided statistics.

        Args:
            stats (dict): A dictionary containing raw scores for various metrics,
                          e.g., {"aum": [...], "el2n": [...], ...}.
        """
        self.ranked_indices_by_method = {}
        self._rank(stats)

    def get_ranked_indices(self) -> dict:
        """Returns the dictionary of ranked indices for all processed metrics."""
        return self.ranked_indices_by_method

    def _rank(self, stats: dict):
        """Routes statistics to the appropriate ranking method."""
        if "aum" in stats:
            self.ranked_indices_by_method["aum"] = self._rank_by_aum(stats["aum"])
        if "datamap" in stats:
            self.ranked_indices_by_method["datamap"] = self._rank_by_datamap(stats["datamap"])
        if "el2n" in stats:
            self.ranked_indices_by_method["el2n"] = self._rank_by_el2n(stats["el2n"])
        if "grand" in stats:
            self.ranked_indices_by_method["grand"] = self._rank_by_grand(stats["grand"])
        if "loss" in stats:
            self.ranked_indices_by_method["loss"] = self._rank_by_loss(stats["loss"])
        if "forgetting" in stats:
            self.ranked_indices_by_method["forgetting"] = self._rank_by_forgetting(stats["forgetting"])
        if "predictions" in stats and "true_labels" in stats:
             accuracy_scores = (np.array(stats['predictions']) == np.array(stats['true_labels'])).astype(int)
             self.ranked_indices_by_method["accuracy"] = self._rank_by_accuracy(accuracy_scores)

    def _rank_by_aum(self, aum_scores: list) -> list:
        """Ranks by AUM, where lower scores are considered harder."""
        ranked_indices_over_epochs = []
        for epoch_scores in aum_scores:
            scores = np.array(epoch_scores, dtype=float)
            scores[np.isnan(scores)] = -np.inf # Treat NaNs as infinitely hard.
            ranked_indices = np.argsort(-scores) # Sort descending (easy to hard).
            ranked_indices_over_epochs.append(ranked_indices)
        return ranked_indices_over_epochs

    def _rank_by_el2n(self, el2n_scores: list) -> list:
        """Ranks by EL2N, where higher scores are considered harder."""
        ranked_indices_over_epochs = []
        for epoch_scores in el2n_scores:
            scores = np.array(epoch_scores, dtype=float)
            scores[np.isnan(scores)] = np.inf # Treat NaNs as infinitely hard.
            ranked_indices = np.argsort(scores) # Sort ascending (easy to hard).
            ranked_indices_over_epochs.append(ranked_indices)
        return ranked_indices_over_epochs

    def _rank_by_grand(self, grand_scores: list) -> list:
        """Ranks by GraND, where higher scores are considered harder."""
        return self._rank_by_el2n(grand_scores)

    def _rank_by_loss(self, loss_scores: list) -> list:
        """Ranks by loss, where higher loss is considered harder."""
        ranked_indices_over_epochs = []
        for epoch_loss_obj_array in loss_scores:
            scores = np.array([item[0] if item else np.nan for item in epoch_loss_obj_array], dtype=float)
            scores[np.isnan(scores)] = np.inf # Treat NaNs as infinitely hard.
            ranked_indices = np.argsort(scores)
            ranked_indices_over_epochs.append(ranked_indices)
        return ranked_indices_over_epochs

    def _rank_by_datamap(self, datamap_stats: dict) -> list:
        """
        Ranks by Data Map confidence and variability.
        Low confidence and high variability are harder. A composite score of
        (variability - confidence) is used for ranking.
        """
        confidence_scores = datamap_stats['confidence']
        variability_scores = datamap_stats['variability']
        ranked_indices_over_epochs = []

        for conf_epoch, var_epoch in zip(confidence_scores, variability_scores):
            conf = np.array(conf_epoch, dtype=float)
            var = np.array(var_epoch, dtype=float)
            
            composite_score = var - conf # Lower is easier (high confidence, low variability).
            composite_score[np.isnan(composite_score)] = np.inf
            
            ranked_indices = np.argsort(composite_score)
            ranked_indices_over_epochs.append(ranked_indices)
        return ranked_indices_over_epochs

    def _rank_by_forgetting(self, forgetting_scores: list) -> list:
        """
        Ranks by forgetting events, where more forgetting is harder.
        Samples that were 'never learned' (-1) are considered the hardest.
        """
        ranked_indices_over_epochs = []
        for forgetting_counts in forgetting_scores:
            scores = np.array(forgetting_counts, dtype=float)
            # Find a value larger than any real count to represent 'never learned'.
            hardest_val = np.max(scores[scores != -1]) + 1 if np.any(scores != -1) else 1
            scores[scores == -1] = hardest_val
            
            ranked_indices = np.argsort(scores)
            ranked_indices_over_epochs.append(ranked_indices)
        return ranked_indices_over_epochs

    def _rank_by_accuracy(self, accuracy_scores: np.ndarray) -> list:
        """Ranks by cumulative accuracy over epochs, where lower accuracy is harder."""
        ranked_indices_over_epochs = []
        for epoch_idx in range(len(accuracy_scores)):
            # Calculate mean accuracy up to the current epoch.
            cumulative_data = accuracy_scores[:epoch_idx + 1]
            avg_accuracy = np.nanmean(cumulative_data, axis=0)
            avg_accuracy[np.isnan(avg_accuracy)] = -np.inf # Treat NaNs as infinitely hard.
            
            ranked_indices = np.argsort(-avg_accuracy) # Sort descending (easy to hard).
            ranked_indices_over_epochs.append(ranked_indices)
        return ranked_indices_over_epochs
