import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class Evaluator:
    def __init__(self, total_samples, stats, percentile=80):
        self.total_samples = total_samples
        self.scores_by_method = {}
        self.percentile = percentile

        # Storage for binary difficulty labels
        self.binary_scores = {}
        self._evaluate(stats)

    def _evaluate(self, stats):
        # AUM
        if "aum" in stats and "datamap" in stats:
            aum_scores = stats["aum"]["sample_margins"]
            confidence_scores = stats["datamap"]["confidence"]
            variability_scores = stats["datamap"]["variability"]
            correctness_scores = stats["datamap"]["correctness"]
            self.binary_scores["aum"] = self._binary_from_aum(aum_scores, confidence_scores, variability_scores, correctness_scores)
            self.binary_scores["datamap"] = self._binary_from_datamap(confidence_scores, variability_scores, correctness_scores)

        # EL2N
        if "el2n" in stats:
            el2n_scores = stats["el2n"]
            self.binary_scores["el2n"] = self._binary_from_el2n(el2n_scores)

        # GraNd
        if "grand" in stats:
            grand_scores = stats["grand"]
            self.binary_scores["grand"] = self._binary_from_grand(grand_scores)

        # Loss
        if "loss" in stats:
            loss = stats["loss"]["all_losses"]
            self.binary_scores["loss"] = self._binary_from_loss(loss)

        # Forgetting
        if "forgetting" in stats:
            forgetting = stats["forgetting"]["epoch_history"]
            self.binary_scores["forgetting"] = self._binary_from_forgetting(forgetting)

        # Regularisation
        if "regularisation" in stats:
            self.binary_scores["regularisation"] = stats["regularisation"]

        if "accuracy" in stats:
            accuracy = stats["accuracy"]
            # all_accuracies = []
            # for epoch_preds in stats['predictions']:
            #     correct = (np.array(epoch_preds) == np.array(stats['true_labels'])).astype(float)
            #     all_accuracies.append(correct)
            self.binary_scores["accuracy"] = self._binary_from_accuracy(accuracy)

    def _binary_from_aum(self, aum_scores, confidence_scores, variability_scores, correctness_scores):
        if any(x is None for x in [aum_scores, confidence_scores, variability_scores, correctness_scores]):
            raise ValueError("Missing one or more required features: AUM, confidence, variability, correctness")

        sample_ids = sorted(aum_scores.keys())
        aum_array = np.array([aum_scores[sid] for sid in sample_ids])
        aum_array = np.transpose(aum_array)

        num_epochs = aum_array.shape[0]
        binary_labels_over_epochs = []

        for epoch_idx in range(num_epochs):
            aum = np.nanmean(aum_array[:epoch_idx + 1], axis=0)

            conf = confidence_scores[epoch_idx]
            var = variability_scores[epoch_idx]
            corr = correctness_scores[epoch_idx]

            features = np.stack([aum, conf, var, corr], axis=1)
            valid_mask = ~np.isnan(features).any(axis=1)
            valid_features = features[valid_mask]

            scaler = StandardScaler()
            norm_features = scaler.fit_transform(valid_features)

            gmm = GaussianMixture(n_components=3, random_state=42)
            cluster_labels = gmm.fit_predict(norm_features)

            conf_valid = conf[valid_mask]
            cluster_conf_means = [conf_valid[cluster_labels == i].mean() for i in range(3)]
            hard_cluster = int(np.argmin(cluster_conf_means))

            binary_valid = (cluster_labels == hard_cluster).astype(int)

            binary_labels = np.zeros(len(conf), dtype=int)
            binary_labels[valid_mask] = binary_valid
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs

    def _binary_from_el2n(self, el2n_scores):
        num_epochs = len(el2n_scores)
        el2n_array = np.array(el2n_scores)
        binary_labels_over_epochs = []

        for epoch_idx in range(num_epochs):
            current_scores = el2n_array[epoch_idx]
            valid_scores = current_scores[~np.isnan(current_scores)]
            threshold = np.percentile(valid_scores, self.percentile)

            binary_labels = np.where(current_scores >= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs


    def _binary_from_grand(self, grand_scores):
        num_epochs = len(grand_scores)
        grand_array = np.array(grand_scores)
        binary_labels_over_epochs = []

        for epoch_idx in range(num_epochs):
            current_scores = grand_array[epoch_idx]
            valid_scores = current_scores[~np.isnan(current_scores)]
            threshold = np.percentile(valid_scores, self.percentile)

            binary_labels = np.where(current_scores >= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs

    def _binary_from_loss(self, loss_scores):
        num_epochs = len(loss_scores)
        loss_array = np.array(loss_scores)
        binary_labels_over_epochs = []

        for epoch_idx in range(num_epochs):
            current_loss = loss_array[epoch_idx]
            valid_scores = current_loss[~np.isnan(current_loss)]
            threshold = np.percentile(valid_scores, self.percentile)

            binary_labels = np.where(current_loss >= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs

    def _binary_from_datamap(self, confidence_scores, variability_scores, correctness_scores):
        if any(x is None for x in [confidence_scores, variability_scores, correctness_scores]):
            raise ValueError("Missing one or more required Data Maps features.")

        confidence_scores = np.array(confidence_scores)
        variability_scores = np.array(variability_scores)
        correctness_scores = np.array(correctness_scores)
        num_epochs = confidence_scores.shape[0]
        binary_labels_over_epochs = []

        for epoch_idx in range(num_epochs):
            conf = confidence_scores[epoch_idx]
            var = variability_scores[epoch_idx]
            corr = correctness_scores[epoch_idx]

            valid_mask = ~np.isnan(conf) & ~np.isnan(var) & ~np.isnan(corr)
            num_valid = valid_mask.sum()
            binary_labels = np.ones_like(conf, dtype=int)  # Default all to hard (1)

            if num_valid == 0:
                binary_labels_over_epochs.append(binary_labels)
                continue

            # High confidence, low variability = easier
            # We subtract var to make lower variability increase the score
            ease_score = conf[valid_mask] - var[valid_mask]

            # Get threshold to label bottom 33% as easy
            threshold = np.percentile(ease_score, 33)
            easy_indices = np.where(valid_mask)[0][ease_score <= threshold]
            binary_labels[easy_indices] = 0
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs

    def _binary_from_forgetting(self, forgetting):
        binary_labels_over_epochs = []

        for forgetting_counts in forgetting:
            counts = np.array(forgetting_counts)
            # Hard if never learned (-1) or forgotten (>=1)
            binary_labels = np.where((counts == -1) | (counts >= 1), 1, 0)
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs

    def _binary_from_regularisation(self):
        return self.scores_by_method.get("regularisation")

    def _binary_from_accuracy(self, accuracy_scores):
        num_epochs = len(accuracy_scores)
        accuracy_array = np.array(accuracy_scores)  # Shape: [num_epochs, num_samples]
        binary_labels_over_epochs = []

        for epoch_idx in range(num_epochs):
            current_scores = accuracy_array[:epoch_idx + 1]
            avg_accuracy = np.mean(current_scores, axis=0)
            valid_scores = avg_accuracy[~np.isnan(avg_accuracy)]
            threshold = np.percentile(valid_scores, 100 - self.percentile)  # Low accuracy = hard

            binary_labels = np.where(avg_accuracy <= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs

