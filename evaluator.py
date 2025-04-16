import numpy as np

class Evaluator:
    def __init__(self, total_samples, stats):
        self.total_samples = total_samples
        self.scores_by_method = {}
        self.percentile = 80
        self._parse_stats(stats)

    def _parse_stats(self, stats):
        # AUM
        if "aum" in stats and "sample_margins" in stats["aum"]:
            aum = stats["aum"]["sample_margins"]
            num_epochs = max(len(v) for v in aum.values())
            arr = np.full((num_epochs, self.total_samples), np.nan)
            for sample_id, values in aum.items():
                arr[:len(values), sample_id] = values
            self.scores_by_method["aum"] = arr

        # EL2N
        if "el2n" in stats:
            el2n = np.array(stats["el2n"])
            self.scores_by_method["el2n"] = el2n

        # GraNd
        if "grand" in stats:
            grand = np.array(stats["grand"])
            self.scores_by_method["grand"] = grand

        # Loss
        if "loss" in stats and "all_losses" in stats["loss"]:
            loss = np.array(stats["loss"]["all_losses"])
            self.scores_by_method["loss"] = loss

        # DataMap
        if "datamap" in stats:
            dm = stats["datamap"]
            if "confidence" in dm:
                conf = np.array(dm["confidence"])
                self.scores_by_method["confidence"] = self._repeat_across_epochs(conf)
            if "variability" in dm:
                var = np.array(dm["variability"])
                self.scores_by_method["variability"] = self._repeat_across_epochs(var)
            if "correctness" in dm:
                corr = np.array(dm["correctness"])
                self.scores_by_method["correctness"] = self._repeat_across_epochs(corr)

        # Forgetting
        if "forgetting" in stats and "forgetting_events" in stats["forgetting"]:
            forgetting = stats["forgetting"]["forgetting_events"]
            arr = np.full((1, self.total_samples), np.nan)
            for idx, val in forgetting.items():
                arr[0, idx] = val if val >= 0 else np.nan
            self.scores_by_method["forgetting"] = arr

        # Regularisation
        if "regularisation" in stats:
            regular = np.array(stats["regularisation"])
            self.scores_by_method["regularisation"] = self._repeat_across_epochs(regular)

    def get_binary_difficulty(self, method_name):
        method_func = getattr(self, f"_binary_from_{method_name}", None)
        if method_func is None:
            raise NotImplementedError(f"No binary difficulty logic implemented for method: {method_name}")
        return method_func()

    def _binary_from_aum(self):
        aum_scores = self.scores_by_method.get("aum")
        raise NotImplementedError(f"Not implemented")

    def _binary_from_el2n(self):
        el2n_scores = self.scores_by_method.get("el2n")
        num_epochs = len(el2n_scores)
        el2n_array = np.array(el2n_scores)
        binary_labels_over_epochs = []

        for epoch_idx in range(num_epochs):
            # Mean across epochs 0 to epoch_idx
            mean_scores = np.nanmean(el2n_array[:epoch_idx + 1], axis=0)
            valid_scores = mean_scores[~np.isnan(mean_scores)]
            threshold = np.percentile(valid_scores, self.percentile)

            binary_labels = np.where(mean_scores >= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs

    def _binary_from_grand(self):
        grand_scores = self.scores_by_method.get("grand")
        num_epochs = len(grand_scores)
        grand_array = np.array(grand_scores)
        binary_labels_over_epochs = []

        for epoch_idx in range(num_epochs):
            # Mean across epochs 0 to epoch_idx
            mean_scores = np.nanmean(grand_array[:epoch_idx + 1], axis=0)
            valid_scores = mean_scores[~np.isnan(mean_scores)]
            threshold = np.percentile(valid_scores, self.percentile)

            binary_labels = np.where(mean_scores >= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs

    def _binary_from_loss(self):
        loss_scores = self.scores_by_method.get("loss")
        num_epochs = len(loss_scores)
        loss_array = np.array(loss_scores)
        binary_labels_over_epochs = []

        for epoch_idx in range(num_epochs):
            # Mean across epochs 0 to epoch_idx
            mean_scores = np.nanmean(loss_array[:epoch_idx + 1], axis=0)
            valid_scores = mean_scores[~np.isnan(mean_scores)]
            threshold = np.percentile(valid_scores, self.percentile)

            binary_labels = np.where(mean_scores >= threshold, 1, 0)
            binary_labels_over_epochs.append(binary_labels)

        return binary_labels_over_epochs

    def _binary_from_datamaps(self):
        confidence_scores = np.array(self.scores_by_method.get("confidence"))  # shape [epochs, samples]
        variability_scores = np.array(self.scores_by_method.get("variability"))  # same shape
        raise NotImplementedError(f"Not implemented")

    def _binary_from_forgetting(self):
        forgetting = self.scores_by_method.get("forgetting")
        binary_labels_over_epochs = []
        for forgetting_per_epoch in forgetting:
            binary_labels_over_epochs.append(np.array(forgetting_per_epoch > 0))
        return binary_labels_over_epochs

    # Probably don't need this as scores are already 'binarised'
    def _binary_from_regularisation(self):
        # Misclassified examples are hard
        regular = self.scores_by_method.get("regularisation")
        raise NotImplementedError(f"Not implemented")
