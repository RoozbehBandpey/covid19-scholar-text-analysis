# Create infrastructure using terraform

The easiest way to create all required Azure resources (Resource Group, Azure ML Workspace, Container Registry, and others) is to use the **Infrastructure as Code (IaC)** [pipeline with ARM templates](../environment_setup/iac-create-environment-pipeline-arm.yml) or the [pipeline with Terraform templates](../environment_setup/iac-create-environment-pipeline-tf.yml). The pipeline takes care of setting up all required resources based on these [Azure Resource Manager templates](../environment_setup/arm-templates/cloud-environment.json), or based on these [Terraform templates](../environment_setup/tf-templates).


### Create the IaC Pipeline

In your Azure DevOps project, create a build pipeline from your forked repository:

![Build connect step](./images/build-connect.png)

Select the **Existing Azure Pipelines YAML file** option and set the path to [/environment_setup/iac-create-environment-pipeline-arm.yml](../environment_setup/iac-create-environment-pipeline-arm.yml) or to [/environment_setup/iac-create-environment-pipeline-tf.yml](../environment_setup/iac-create-environment-pipeline-tf.yml), depending on if you want to deploy your infrastructure using ARM templates or Terraform:

![Configure step](./images/select-iac-pipeline.png)

If you decide to use Terraform, make sure the ['Terraform Build & Release Tasks' from Charles Zipp](https://marketplace.visualstudio.com/items?itemName=charleszipp.azure-pipelines-tasks-terraform) is installed.

Having done that, run the pipeline: