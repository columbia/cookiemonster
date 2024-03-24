from systemx.data.criteo.creators.three_advertisers_dataset_creator import ThreeAdversitersDatasetCreator
from systemx.data.criteo.creators.partner_value_dataset_creator import PartnerValueDatasetCreator
from systemx.data.criteo.creators.partner_count_dataset_creator import PartnerCountDatasetCreator

registered_dataset_creators = {
    "three-advertisers": ThreeAdversitersDatasetCreator,
    "partner-values": PartnerValueDatasetCreator,
    "partner-counts": PartnerCountDatasetCreator,
}
