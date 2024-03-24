from typer import Typer

from systemx.data.criteo.creators import ThreeAdversitersDatasetCreator, DatasetCreator
    
app = Typer()

@app.command()
def create_dataset(
    datasettype: str = "vanilla"
):
    if datasettype == "three-advertisers":
        creator = ThreeAdversitersDatasetCreator()
    elif datasettype == "vanilla":
        creator = DatasetCreator()
    else:
        raise NotImplementedError(f"dataset creator of type {dataset_type} has not been implemented")
    
    creator.create_datasets()

if __name__ == "__main__":
    app()
