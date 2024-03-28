from typer import Typer

from cookiemonster.data.criteo.creators import registered_dataset_creators

app = Typer()


@app.command()
def create_dataset(datasettype: str = "partner-values"):
    dataset_creator = registered_dataset_creators.get(datasettype)
    if not dataset_creator:
        raise NotImplementedError(
            f"dataset creator of type {datasettype} has not been implemented. Supported dataset types are {registered_dataset_creators.keys()}"
        )

    dataset_creator().create_datasets()


if __name__ == "__main__":
    app()