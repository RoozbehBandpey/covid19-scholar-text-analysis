# Configure the Azure provider
terraform {
  required_providers {
    azurerm = {
      source = "hashicorp/azurerm"
      version = ">= 2.26"
    }
  }
}

provider "azurerm" {
  features {}
}

variable BASE_NAME {
	type = string
	default = "azure-search"
}
variable ENV {
	type = string
	default = "poc"
}
variable LOCATION {
	type = string
	default = "westeurope"
}

resource "azurerm_resource_group" "rg" {
  name     = "${var.BASE_NAME}-rg-${var.ENV}"
  location = var.LOCATION
}