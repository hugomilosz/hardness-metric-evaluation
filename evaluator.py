import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class Evaluator:
    def __init__(self, total_samples, stats, percentile=80):
        self.total_samples = total_samples
        self.scores_by_method = {}
        self.binary_scores = {}
        self._evaluate(stats)

    def _evaluate(self, stats):
        if "aum" in stats and "datamap" in stats:
            self.binary_scores["aum"] = self._binary_from_aum(stats["aum"])
            self.binary_scores["datamap"] = self._binary_from_datamap(stats["datamap"])
        if "el2n" in stats:
            self.binary_scores["el2n"] = self._binary_from_el2n(stats["el2n"])
        if "grand" in stats:
            self.binary_scores["grand"] = self._binary_from_grand(stats["grand"])
        if "loss" in stats:
            self.binary_scores["loss"] = self._binary_from_loss(stats["loss"])
        if "forgetting" in stats:
            self.binary_scores["forgetting"] = self._binary_from_forgetting(stats["forgetting"])
        if "regularisation" in stats:
            self.binary_scores["regularisation"] = stats["regularisation"]
        if "predictions" in stats and "true_labels" in stats:
            predictions = np.array(stats['predictions'])
            true_labels = np.array(stats['true_labels'])
            accuracy_scores = (predictions == true_labels).astype(int)
            self.binary_scores["accuracy"] = self._binary_from_accuracy(accuracy_scores)

    # 67% examples are labelled as HARD (Follows Dataset Cartograpy)
    def _binary_from_aum(self, aum_scores):
        binary_labels_over_epochs = []
        for epoch_scores in aum_scores:
            current_scores = np.array(epoch_scores, dtype=float)
            if current_scores.size == 0: continue
            valid_scores = current_scores[~np.isnan(current_scores)]
            if valid_scores.size == 0:
                binary_labels_over_epochs.append(np.zeros_like(current_scores, dtype=int))
                continue
            # Labels HARD and AMBIGUOUS examples as hard - below 67th percentile
            threshold = np.percentile(valid_scores, 100*(2/3.0))
            binary_labels = np.where(current_scores <= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)
        return np.array(binary_labels_over_epochs, dtype=object)

    # 30% examples are labelled as HARD (as mentioned in paper)
    def _binary_from_el2n(self, el2n_scores):
        binary_labels_over_epochs = []
        for epoch_scores in el2n_scores:
            current_scores = np.array(epoch_scores, dtype=float)
            if current_scores.size == 0: continue
            valid_scores = current_scores[~np.isnan(current_scores)]
            if valid_scores.size == 0:
                binary_labels_over_epochs.append(np.zeros_like(current_scores, dtype=int)); continue
            threshold = np.percentile(valid_scores, 70)
            binary_labels = np.where(current_scores >= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)
        return np.array(binary_labels_over_epochs)

    # 30% examples are labelled as HARD (as mentioned in paper)
    def _binary_from_grand(self, grand_scores):
        binary_labels_over_epochs = []
        for epoch_scores in grand_scores:
            current_scores = np.array(epoch_scores, dtype=float)
            if current_scores.size == 0: continue
            valid_scores = current_scores[~np.isnan(current_scores)]
            if valid_scores.size == 0:
                binary_labels_over_epochs.append(np.zeros_like(current_scores, dtype=int)); continue
            threshold = np.percentile(valid_scores, 70)
            binary_labels = np.where(current_scores >= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)
        return np.array(binary_labels_over_epochs)

    # 30% examples are labelled as HARD
    def _binary_from_loss(self, loss_scores):
        binary_labels_over_epochs = []
        for epoch_loss_obj_array in loss_scores:
            try:
                current_loss = np.array([item[0] if item else np.nan for item in epoch_loss_obj_array], dtype=float)
                
                if current_loss.size == 0: continue
                valid_scores = current_loss[~np.isnan(current_loss)]
                if valid_scores.size == 0:
                    binary_labels_over_epochs.append(np.zeros_like(current_loss, dtype=int)); continue
                
                threshold = np.percentile(valid_scores, 70)
                binary_labels = np.where(current_loss >= threshold, 1, 0)
                binary_labels_over_epochs.append(binary_labels)
            except (TypeError, IndexError) as e:
                print(f"Warning: Could not parse loss epoch data due to structure issue: {e}. Skipping epoch.")
                continue
        return np.array(binary_labels_over_epochs)

    # 67% examples are labelled as HARD
    def _binary_from_datamap(self, datamap_stats):
        """
        Labels hard-to-learn AND ambiguous examples as hard
        """
        confidence_scores = datamap_stats['confidence']
        variability_scores = datamap_stats['variability']
        
        binary_labels_over_epochs = []
        for epoch_idx in range(len(confidence_scores)):
            conf = np.array(confidence_scores[epoch_idx], dtype=float)
            var = np.array(variability_scores[epoch_idx], dtype=float)

            binary_labels = np.ones_like(conf, dtype=int)

            variability_threshold = np.percentile(var, 100 * (2/3.0))
            low_variability_mask = var < variability_threshold

            low_var_confidence_scores = conf[low_variability_mask]
            if low_var_confidence_scores.size == 0:
                binary_labels_over_epochs.append(binary_labels)
                continue
            
            confidence_threshold = np.median(low_var_confidence_scores)
            hard_to_learn_mask = low_variability_mask & (conf > confidence_threshold)
            binary_labels[hard_to_learn_mask] = 0
            
            binary_labels_over_epochs.append(binary_labels)
            
        return np.array(binary_labels_over_epochs)

    # Forgotten or Never Learned Examples labelled as HARD
    def _binary_from_forgetting(self, forgetting_scores):
        binary_labels_over_epochs = []
        for forgetting_counts in forgetting_scores:
            counts = np.array(forgetting_counts)
            binary_labels = np.where((counts == -1) | (counts >= 1), 1, 0)
            binary_labels_over_epochs.append(binary_labels)
        return np.array(binary_labels_over_epochs)

    # 30% examples are labelled as HARD
    def _binary_from_accuracy(self, accuracy_scores):
        binary_labels_over_epochs = []
        for epoch_idx, epoch_data in enumerate(accuracy_scores):
            # Process cumulative accuracy up to the current epoch
            cumulative_data = accuracy_scores[:epoch_idx + 1]
            max_len = max(len(row) for row in cumulative_data)
            padded_data = np.full((len(cumulative_data), max_len), np.nan, dtype=float)
            for i, row in enumerate(cumulative_data):
                padded_data[i, :len(row)] = row
            avg_accuracy = np.nanmean(padded_data, axis=0)
            valid_scores = avg_accuracy[~np.isnan(avg_accuracy)]
            if valid_scores.size == 0:
                binary_labels_over_epochs.append(np.zeros_like(avg_accuracy, dtype=int)); continue
            threshold = np.percentile(valid_scores, 30)
            binary_labels = np.where(avg_accuracy <= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)
        return np.array(binary_labels_over_epochs)