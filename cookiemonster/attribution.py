from cookiemonster.report import HistogramReport, Report, ScalarReport


class AttributionFunction:

    def __init__(self, sensitivity_metric) -> None:
        self.sensitivity_metric = sensitivity_metric

    def compute_global_sensitivity(self):
        pass

    def compute_individual_sensitivity(self, partition, epoch_id):
        """
        We only look at the partition metadata, and the events from epoch_id.
        """
        pass

    def create_empty_report(self) -> Report:
        pass

    def create_report(self, partition, **kwargs) -> Report:
        pass

    # TODO: maybe add null_report too


class LastTouch(AttributionFunction):

    def __init__(self, sensitivity_metric, attribution_cap) -> None:
        self.attribution_cap = attribution_cap

        if sensitivity_metric != "L1":
            raise NotImplementedError(
                f"Unsupported sensitivity metric: {sensitivity_metric}"
            )

        super().__init__(sensitivity_metric)

    def compute_global_sensitivity(self):
        return self.attribution_cap

    def compute_individual_sensitivity(self, partition, epoch_id):
        if partition.epochs_window_size() == 1:
            if partition.unbiased_report is None:
                raise ValueError("You need to create the unbiased report first")

            return sum(list(partition.unbiased_report.histogram.values()))

        if epoch_id not in partition.impressions_per_epoch:
            # Epoch not in the attribution window
            return 0

        if not partition.impressions_per_epoch[epoch_id]:
            # Epoch with no relevant impressions
            return 0

        return self.attribution_cap

    def create_empty_report(self):
        return ScalarReport()

    def create_report(self, partition, filter, key_piece: str):

        report = ScalarReport()

        # Scan all impressions in epochs and keep the latest one
        epochs = sorted(list(partition.impressions_per_epoch.keys()), reverse=True)
        for epoch in epochs:
            impressions = partition.impressions_per_epoch[epoch]

            if impressions:
                impression_key = impressions[-1].key
                if impression_key == "nan":
                    impression_key = ""

                # Sort impression keys and stringify them
                bucket_key = impression_key + "#" + filter + "#" + key_piece
                bucket_value = self.value

                report.add(bucket_key, bucket_value)
                break

        if report.empty():
            bucket_key = "#" + filter + "#" + key_piece
            bucket_value = 0
            report.add(bucket_key, bucket_value)

        return report


class LastTouchWithCount(AttributionFunction):
    """
    Keep a default bucket to count epochs with no impressions (i.e. relevant events)
    """

    def __init__(self, sensitivity_metric, attribution_cap, kappa) -> None:
        self.attribution_cap = attribution_cap

        assert isinstance(kappa, float) or isinstance(kappa, int)
        self.kappa = kappa

        if sensitivity_metric != "L1":
            raise NotImplementedError(
                f"Unsupported sensitivity metric: {sensitivity_metric}"
            )

        super().__init__(sensitivity_metric)

    def compute_global_sensitivity(self):
        return max(self.attribution_cap, self.kappa)

    def compute_individual_sensitivity(self, partition):
        pass

    def create_empty_report(self):
        return HistogramReport(impression_buckets=["empty", "main"])

    def create_report(self, partition, filter, key_piece: str):

        report = HistogramReport(impression_buckets=["empty", "main"])
        already_attributed = False

        # Browse all the epochs, even those with no impressions
        (x, y) = partition.epochs_window
        for epoch in range(y, x - 1, -1):
            impressions = partition.impressions_per_epoch.get(epoch, [])

            if impressions and not already_attributed:
                # For epochs with impressions but already_attributed, there is no bias
                impression_key = impressions[-1].key
                if impression_key == "nan":
                    impression_key = "main"

                # Sort impression keys and stringify them
                bucket_key = impression_key + "#" + filter + "#" + key_piece
                bucket_value = self.attribution_cap
                report.add(bucket_key, bucket_value)

                already_attributed = True
            else:
                # This epoch has no impressions.
                # Maybe it is really the case, or maybe it got zeroed-out by a filter
                # TODO: add a heartbeat here
                default_bucket_prefix = "empty"
                bucket_key = default_bucket_prefix + "#" + filter + "#" + key_piece
                bucket_value = self.kappa
                report.add(bucket_key, bucket_value)

        return report
