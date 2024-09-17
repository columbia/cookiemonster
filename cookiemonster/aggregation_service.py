from abc import abstractmethod
from enum import Enum

import numpy as np

from cookiemonster.query_batch import QueryBatch


class AggregationServiceType(str, Enum):
    LOCAL_LAPLACIAN = "local_laplacian"
    REMOTE_LAPLACIAN = "remote_laplacian"


class AggregationResult:
    def __init__(
        self,
        true_output: float,
        aggregation_output: float,
        aggregation_noisy_output: float,
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
        """Isotropic Laplace noise, i.e. iid Laplace on each coordinate"""
        true_output = sum(query_batch.unbiased_values)
        aggregation_output = sum(query_batch.values)
        noise_scale = query_batch.noise_scale
        # noise_scale = query_batch.global_sensitivity / query_batch.global_epsilon
        
        if isinstance(aggregation_output, np.ndarray):
            noise = np.random.laplace(scale=noise_scale, size=aggregation_output.shape)
        else:
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
        raise NotImplementedError("RemoteLaplacianAggregationService is not implemented yet")
