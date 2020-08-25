# Check core SDK version number
import azureml.core
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace, Datastore, Dataset, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
import os

print("SDK version:", azureml.core.VERSION)


ws = Workspace.from_config(
    auth=AzureCliAuthentication(),
    path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'config.json'
    )
)

print(f'Workspace: {ws.name}')


experiment_name = 'train-bert-ner-on-amlcompute'
# experiment = Experiment(workspace=ws, name=experiment_name)


supported_vms = AmlCompute.supported_vmsizes(workspace=ws)
print(supported_vms)


project_folder = './ner'


bert_env = Environment("bert_aml_env")



bert_env.docker.enabled = True
bert_env.from_conda_specification('bert', 'environment.yml')

bert_env.register(ws)
