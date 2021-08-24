from argoflow.tasks import *
from argoflow.workflow import *

tasks = taskFlow()

tasks.addJob(
    name="generateSyntheticData",
    template="synthea",
    parameters=[{"name": "population", "value": "10"}],
)

tasks.addJob(
    name="transformData",
    template="pyspark",
    parameters=[
        {"name": "observations", "value": "mounted/data/observations.csv"},
        {"name": "patients", "value": "mounted/data/patients.csv"},
        {"name": "conditions", "value": "mounted/data/conditions.csv"},
        {"name": "target", "value": "mounted/data/processed.parquet"},
    ],
    dependencies=["generateSyntheticData"],
)

tasks.addJob(
    name="model",
    template="scikit",
    parameters=[
        {"name": "source", "value": "data/processed.parquet"},
        {"name": "target", "value": "data/model.pkl"},
    ],
    dependencies=["transformData"],
)

tasks.addJob(
    name="Deploy",
    template="streamit",
    parameters=[{"name": "model", "value": "data/model.pkl"}],
    dependencies=["model"],
)

data = tasks.compile()
dag = workflow("DiabetesModel", data,authpath='./config.yaml')
dag.submit()
