from cookiemonster.data.criteo.creators.three_advertisers_dataset_creator import (
    ThreeAdversitersDatasetCreator,
)
from cookiemonster.data.criteo.creators.partner_value_dataset_creator import (
    PartnerValueDatasetCreator,
)
from cookiemonster.data.criteo.creators.partner_count_dataset_creator import (
    PartnerCountDatasetCreator,
)
from cookiemonster.data.criteo.creators.query_pool_creator import (
    QueryPoolDatasetCreator
)

registered_dataset_creators = {
    "three-advertisers": ThreeAdversitersDatasetCreator,
    "partner-values": PartnerValueDatasetCreator,
    "partner-counts": PartnerCountDatasetCreator,
    "query-pool": QueryPoolDatasetCreator,
}
