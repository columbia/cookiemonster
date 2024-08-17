import os
import gdown
import zipfile
import process_pqt

# gdown.download_folder(
#     "https://drive.google.com/drive/folders/1qt1YVpxIuhmiCbTn4LDi3CDflkcqi_vf", quiet=True)

# if not os.path.exists("publishers"):
#     os.mkdir("publishers")
# if not os.path.exists("advertisers"):
#     os.mkdir("advertisers")

# for file in os.listdir("patcg_dataset"):
#     if file.endswith(".zip") and file.startswith("advertiser_conversions"):
#         zipfile.ZipFile(
#             f"patcg_dataset/{file}").extractall("advertisers")
#     if file.endswith(".zip") and file.startswith("publisher_exposures"):
#         zipfile.ZipFile(
#             f"patcg_dataset/{file}").extractall("publishers")

# print("downloaded data")

# process_pqt.convert_to_csv()
# process_pqt.filter_converters()