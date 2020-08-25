# Check core SDK version number
import azureml.core
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace, Datastore, Dataset, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
import os
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import DEFAULT_CPU_IMAGE




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
experiment = Experiment(workspace=ws, name=experiment_name)


supported_vms = AmlCompute.supported_vmsizes(workspace=ws)
print(supported_vms)


project_folder = './ner'


bert_env = Environment("bert_aml_env")



bert_env.docker.enabled = True
bert_env.from_conda_specification('bert', './environment.yml')

bert_env.register(ws)


# Choose a name for your CPU cluster
gpu_cluster_name = "gpu-compute-bert"

# Verify that cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                           max_nodes=4)
    gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)

gpu_cluster.wait_for_completion(show_output=True)


src = ScriptRunConfig(source_directory=project_folder, script='train.py')

# Set compute target to the one created in previous step
src.run_config.target = gpu_cluster.name

# Set environment
src.run_config.environment = bert_env

run = experiment.submit(config=src)
print(run)


run.wait_for_completion(show_output=True)


print(run.get_metrics())
