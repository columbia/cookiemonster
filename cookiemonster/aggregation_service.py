import numpy as np
from enum import Enum
from abc import abstractmethod

from cookiemonster.query_batch import QueryBatch


class AggregationServiceType(str, Enum):
    LOCAL_LAPLACIAN = "local_laplacian"
    REMOTE_LAPLACIAN = "remote_laplacian"


class AggregationResult:
    def __init__(
        self, true_output, aggregation_output, aggregation_noisy_output
    ) -> None:
        self.true_output = true_output
        self.aggregation_output = aggregation_output
        self.aggregation_noisy_output = aggregation_noisy_output
        # self.bias = abs(true_output - aggregation_output)


class AggregationService:
    @classmethod
    def create(cls, aggregation_service: str) -> "AggregationService":
        if aggregation_service == AggregationServiceType.LOCAL_LAPLACIAN:
            return LocalLaplacianAggregationService()
        elif aggregation_service == AggregationServiceType.REMOTE_LAPLACIAN:
            return RemoteLaplacianAggregationService()
        else:
            raise ValueError("No support for the requested aggregation service")

    @abstractmethod
    def create_summary_report(self, query_batch: QueryBatch) -> AggregationResult:
        pass


class LocalLaplacianAggregationService(AggregationService):

    def create_summary_report(self, query_batch: QueryBatch) -> AggregationResult:
        true_output = sum(query_batch.unbiased_values)
        aggregation_output = sum(query_batch.values)

        noise_scale = query_batch.global_sensitivity / query_batch.global_epsilon
        noise = np.random.laplace(scale=noise_scale)
        aggregation_noisy_output = aggregation_output + noise
        return AggregationResult(
            true_output, aggregation_output, aggregation_noisy_output
        )


class RemoteLaplacianAggregationService(LocalLaplacianAggregationService):
    """
    RemoteLaplacianAggregationService - aggregation service that delegates summary report aggregation to remote processes
    """

    def create_summary_report(self, query_batch: QueryBatch) -> AggregationResult:
        # TODO: send out to create summary report
        return super().create_summary_report()
