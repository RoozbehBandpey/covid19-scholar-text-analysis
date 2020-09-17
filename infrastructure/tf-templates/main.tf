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
variable SKU {
	type = string
  	description = "Valid values are 'free', 'standard', 'standard2', and 'standard3' (2 & 3 must be enabled on the backend by Microsoft support). 'free' provisions the service in shared clusters. 'standard' provisions the service in dedicated clusters."
  	default     = "standard2"
}

variable REPLICA_COUNT {
	type = string
  	description = "Replicas distribute search workloads across the service. You need 2 or more to support high availability (applies to Basic and Standard only)."
  	default     = 3
}

variable PARTITION_COUNT {
	type = string
  	description = "Partitions allow for scaling of document count as well as faster indexing by sharding your index over multiple Azure Search units. Allowed values: 1, 2, 3, 4, 6, 12"
  	default     = 1
}

variable HOSTING_MODE {
	type = string
  	description = "Applicable only for SKU set to standard3. You can set this property to enable a single, high density partition that allows up to 1000 indexes, which is much higher than the maximum indexes allowed for any other SKU. Allowed values: default, highDensity"
  	default     = "Default"
}

resource "azurerm_resource_group" "rg" {
  	name     = "${var.BASE_NAME}-rg-${var.ENV}"
  	location = "${var.LOCATION}"
}

resource "azurerm_search_service" "search" {
  	name                = "${var.BASE_NAME}-as-${var.ENV}"
  	resource_group_name = "${azurerm_resource_group.rg.name}"
  	location            = "${var.LOCATION}"
  	sku                 = "${var.SKU}"
  	replica_count       = "${var.REPLICA_COUNT}"
  	partition_count     = "${var.PARTITION_COUNT}"
}