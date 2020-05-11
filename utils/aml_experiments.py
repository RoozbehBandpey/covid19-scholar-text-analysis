import os
import sys
from pprint import pprint
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import AzureCliAuthentication, MsiAuthentication


class Extractor():

	def __init__(self, ws_config_file):
		self.ws = Workspace.from_config(
			auth=AzureCliAuthentication(),
			path=os.path.join(
				os.path.dirname(os.path.realpath(__file__)),
				ws_config_file
			)
		)
		print(
			f"Found workspace {self.ws.name} \n\tat location atpdeep_cv1_prd_ws.location{self.ws.location}\n\t with the id:{self.ws._workspace_id}")
		self.experiments = {}

	def get_experiments(self):
		for exp_name in self.ws.experiments:
			exp = self.ws.experiments[exp_name]
			if exp_name not in ['WsValidationExp', 'bla']:
				self.experiments[exp_name] = exp
		return self.experiments

	def get_runsIds_per_exp(self, experiment):
		exp_runs = {}
		for run in experiment.get_runs():
			runDetails = run.get_details()
			runId = runDetails.get('runId')
			exp_runs[runId] = experiment
		return exp_runs


if __name__ == "__main__":
	DEBUG = False
	workspace_config_files = [
    	'aml-exp-config.json'
	]
	print("Pyhton versuin")
	print(dir(azureml.core))
	print(f"Azure ML Core version {azureml.core.__version__}")
	extractor = Extractor(workspace_config_files[0])

	experiments = extractor.get_experiments()
	experimentName = "model_inference_cs_demo_video_0_multitask_release_ct"
	experiment = experiments[experimentName]
	pprint(experiment)
	runs = extractor.get_runsIds_per_exp(experiment)
	pprint(runs)
	# for run in runs:
	# 	experiment = runs[run].name
	# 	workspace = runs[run].workspace
	# 	workspace_name = workspace.name
	# 	if DEBUG:
	# 		print(f'Extracting -> {experiment}, {run}, {workspace_name}')
	# 		# metrics = Metrics(workspace=workspace, experiment_name=experiment,
	# 		#                   run_id=run, subscription_id=workspace.subscription_id)
	# 	else:
	# 		metrics = Metrics(workspace=workspace, experiment_name=experiment,
	# 		                  run_id=run, subscription_id=workspace.subscription_id)
	# 		# Fetch and merge
	# 		metrics.merge()
	# 		# Push to SQL DB
	# 		metrics.upload()
	# 		print(f'Inserted -> {experiment}, {run}, {workspace_name}')
