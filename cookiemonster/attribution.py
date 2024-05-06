from typing import List

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

    def sum_reports(self, reports: List[Report]) -> Report:
        """
        Sum all the reports in the list.
        Can be implemented more efficiently by the subclasses.
        """
        report = self.create_empty_report()
        for r in reports:
            report += r
        return report


class LastTouch(AttributionFunction):

    def __init__(self, sensitivity_metric, attribution_cap) -> None:
        # TODO: per-conversion attribution cap (=conversion value) instead of global cap?
        self.attribution_cap = attribution_cap

        if sensitivity_metric != "L1":
            raise NotImplementedError(
                f"Unsupported sensitivity metric: {sensitivity_metric}"
            )

        super().__init__(sensitivity_metric)

    def compute_global_sensitivity(self):
        return self.attribution_cap

    def compute_individual_sensitivity(self, partition, epoch_id):
        if epoch_id not in partition.get_epochs():
            # Epoch not in the attribution window
            return 0

        if partition.epochs_window_size() == 1:
            if partition.unbiased_report is None:
                raise ValueError("You need to create the unbiased report first")

            return sum(list(partition.unbiased_report.histogram.values()))

        if epoch_id not in partition.impressions_per_epoch:
            # Empty epoch (either naturally, or because out-of-budget)
            return 0

        if not partition.impressions_per_epoch[epoch_id]:
            # Epoch with no relevant impressions
            return 0

        return self.compute_global_sensitivity()

    def create_empty_report(self):
        return ScalarReport()

    def create_report(self, partition, filter, key_piece: str):
        
        global_sensitivity = self.compute_global_sensitivity()

        report = ScalarReport(global_sensitivity=global_sensitivity)

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
                bucket_value = partition.value

                report.add(bucket_key, bucket_value)
                break

        # TODO: do we really need this?
        if report.empty():
            bucket_key = "#" + filter + "#" + key_piece
            bucket_value = 0
            report.add(bucket_key, bucket_value)

        return report


class LastTouchWithCount(AttributionFunction):
    """
    Keep a default bucket to count epochs with no impressions (i.e. relevant events)
    `empty` for DP counts, `main` for the scalar count.
    
    We only support single-query reports for now, with potentially multiple buckets
    At aggregation time you need to know how many buckets you are interested in
    Otherwise, the absence or presence of a bucket can leak info about a single record
    Maybe one day add support for arbitrary buckets too, e.g. for histogram queries?
    """

    def __init__(self, sensitivity_metric, attribution_cap, kappa) -> None:

        # TODO: (P!1) attributionacap vs value
        self.attribution_cap = attribution_cap

        # Harcoded buckets.
        self.impression_buckets = ["empty", "main"]

        assert isinstance(kappa, float) or isinstance(kappa, int)
        self.kappa = kappa

        if sensitivity_metric != "L1":
            raise NotImplementedError(
                f"Unsupported sensitivity metric: {sensitivity_metric}"
            )

        super().__init__(sensitivity_metric)

    def compute_global_sensitivity(self):
        return self.kappa + self.attribution_cap

    def compute_individual_sensitivity(self, partition, epoch_id):

        if epoch_id not in partition.get_epochs():
            # Epoch not in the attribution window
            return 0

        if partition.epochs_window_size() == 1:
            if partition.unbiased_report is None:
                raise ValueError("You need to create the unbiased report first")

            # TODO: check kappa here
            return self.kappa + sum(list(partition.unbiased_report.histogram.values()))

        if epoch_id not in partition.impressions_per_epoch:
            # Empty epoch (either naturally, or because out-of-budget)
            return 0

        if not partition.impressions_per_epoch[epoch_id]:
            # Epoch with no relevant impressions, but still has budget!
            return self.kappa

        return self.compute_global_sensitivity()

    def create_empty_report(self):
        return HistogramReport(impression_buckets=self.impression_buckets)

    def create_report(self, partition, filter, key_piece: str):
        
        global_sensitivity = self.compute_global_sensitivity()

        report = HistogramReport(
            global_sensitivity=global_sensitivity,
            impression_buckets=self.impression_buckets)
        already_attributed = False

        # Browse all the epochs, even those with no impressions and the deleted ones
        epochs = partition.get_epochs(reverse=True)
        for epoch in epochs:

            if epoch not in partition.impressions_per_epoch:
                # This epoch belongs to the window but has been erased from the partition
                # That means it has no budget left.
                # We treat it like a totally empty epoch.
                # TODO: double check the maths
                default_bucket_prefix = "empty"
                bucket_key = default_bucket_prefix + "#" + filter + "#" + key_piece
                bucket_value = self.kappa
                report.add(bucket_key, bucket_value)

            else:
                impressions = partition.impressions_per_epoch[epoch]

                if impressions and not already_attributed:
                    impression_key = impressions[-1].key
                    if impression_key == "nan":
                        impression_key = "main"

                    # Sort impression keys and stringify them
                    bucket_key = impression_key + "#" + filter + "#" + key_piece
                    bucket_value = partition.value
                    report.add(bucket_key, bucket_value)

                    already_attributed = True
                if impressions and already_attributed:
                    # We don't need to attribute again. No bias.
                    pass
                else:
                    # This epoch has no impressions, but it still has budget.
                    # We treat it like an epoch that only contains a single heartbeat event.
                    pass

        # For retrocompatibility with the scalar report, but doesn't seem super necessary
        if report.empty():
            default_bucket_prefix = "empty"
            bucket_key = default_bucket_prefix + "#" + filter + "#" + key_piece
            bucket_value = 0
            report.add(bucket_key, bucket_value)

        return report

    def sum_reports(self, reports: List[HistogramReport]) -> HistogramReport:
        
        final_report = HistogramReport(
            impression_buckets=self.impression_buckets
            )
        
        global_sensitivity = 0
        for r in reports:
            # Partitions are disjoint, so we can take the max
            global_sensitivity = max(global_sensitivity, r.global_sensitivity)
            for key, value in r.histogram.items():
                final_report.add(key, value)
                
        final_report.global_sensitivity = global_sensitivity
        
        return final_report
