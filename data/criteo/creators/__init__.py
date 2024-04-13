from data.criteo.creators.query_pool_creator import (
    QueryPoolDatasetCreator,
)
from data.criteo.creators.augmented_impressions_creator import (
    AugmentedImpressionsDatasetCreator,
)

registered_dataset_creators = {
    "query-pool": QueryPoolDatasetCreator,
    "augmented": AugmentedImpressionsDatasetCreator,
}
