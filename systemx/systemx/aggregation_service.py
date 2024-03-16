from abc import abstractmethod
from enum import Enum

class AggregationServiceType(str, Enum):
    LOCAL_NAIVE = "local_naive"
    LOCAL_LAPLACIAN = "local_laplacian"
    REMOTE_NAIVE = "remote_naive"
    REMOTE_LAPLACIAN = "remote_laplacian"

class AggregationService:
    @classmethod
    def create(cls, aggregation_service: str) -> "AggregationService":
        if aggregation_service == AggregationServiceType.LOCAL_LAPLACIAN:
            return LocalLaplacianAggregationService()
        elif aggregation_service == AggregationServiceType.REMOTE_LAPLACIAN:
            return RemoteLaplacianAggregationService()
        elif aggregation_service == AggregationServiceType.REMOTE_NAIVE:
            return RemoteNaiveAggregationService()
        else:
            return LocalNaiveAggregationService()
    
    @abstractmethod
    def create_summary_reports(self, aggregatable_reports: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        pass

class LocalNaiveAggregationService(AggregationService):

    def create_summary_reports(self, aggregatable_reports: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        summary_reports: dict[str, dict[str, float]] = {}
        for destination in aggregatable_reports:
            summary_report: dict[str, float] = {}
            for r in aggregatable_reports[destination]:
                for bucket in r.histogram:
                    if bucket not in summary_report:
                        summary_report[bucket] = 0
                    summary_report[bucket] += r.histogram[bucket]
            summary_reports[destination] = summary_report
        return summary_reports
    
class LocalLaplacianAggregationService(AggregationService):

    def create_summary_reports(self, aggregatable_reports: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        # TODO: we're using L1 sensitivity, but no global epsilon. where will we get the epsilon from?
        raise NotImplementedError()

class RemoteNaiveAggregationService(LocalNaiveAggregationService):
    def create_summary_reports(self, aggregatable_reports: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        # TODO: send out to create summary reports per destination
        return super().create_summary_reports()
    
class RemoteLaplacianAggregationService(LocalLaplacianAggregationService):
    
    def create_summary_reports(self, aggregatable_reports: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        # TODO: send out to create summary reports per destination
        return super().create_summary_reports()
