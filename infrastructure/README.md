# Create infrastructure using terraform

The easiest way to create all required Azure resources (Resource Group, Azure ML Workspace, Container Registry, and others) is to use the **Infrastructure as Code (IaC)** [pipeline with ARM templates](../environment_setup/iac-create-environment-pipeline-arm.yml) or the [pipeline with Terraform templates](../environment_setup/iac-create-environment-pipeline-tf.yml). The pipeline takes care of setting up all required resources based on these [Azure Resource Manager templates](../environment_setup/arm-templates/cloud-environment.json), or based on these [Terraform templates](../environment_setup/tf-templates).


### Create the IaC Pipeline

In your Azure DevOps project, create a build pipeline from your forked repository:

![Build connect step](./images/build-connect.png)

Select the **Existing Azure Pipelines YAML file** option and set the path to [/environment_setup/iac-create-environment-pipeline-arm.yml](../environment_setup/iac-create-environment-pipeline-arm.yml) or to [/environment_setup/iac-create-environment-pipeline-tf.yml](../environment_setup/iac-create-environment-pipeline-tf.yml), depending on if you want to deploy your infrastructure using ARM templates or Terraform:

![Configure step](./images/select-iac-pipeline.png)

If you decide to use Terraform, make sure the ['Terraform Build & Release Tasks' from Charles Zipp](https://marketplace.visualstudio.com/items?itemName=charleszipp.azure-pipelines-tasks-terraform) is installed.

Having done that, run the pipeline:




### Add Service principle
Search for `App registeration` this an azure AD app
Register an app with desired name
Set a secret for the app that never expires
Go to sunscription,in role assignment add a new role assignment
Pick owner and serach for your app name



When programmatically signing in, you need to pass the tenant ID with your authentication request and the application ID. You also need a certificate or an authentication key (described in the following section).

Copy the Application ID and store it in your application code.


Create a new application secret
If you choose not to use a certificate, you can create a new application secret.

Select Azure Active Directory.

From App registrations in Azure AD, select your application.

Select Certificates & secrets.

Select Client secrets -> New client secret.

Provide a description of the secret, and a duration. When done, select Add.

After saving the client secret, the value of the client secret is displayed. Copy this value because you won't be able to retrieve the key later. You will provide the key value with the application ID to sign in as the application. Store the key value where your application can retrieve it.



TODO: add those to keyvault, no more local settings



Set environment variables
In order for Terraform to use the intended Azure subscription, set environment variables. You can set the environment variables at the Windows system level or in within a specific PowerShell session. If you want to set the environment variables for a specific session, use the following code. Replace the placeholders with the appropriate values for your environment.

```bash
$ setx ARM_CLIENT_ID="<service_principal_app_id>"
$ setx ARM_SUBSCRIPTION_ID="<azure_subscription_id>"
$ setx ARM_TENANT_ID="<azure_subscription_tenant_id>"
```
