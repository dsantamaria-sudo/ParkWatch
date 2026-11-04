// ParkWatch — Azure ML infrastructure.
// Workspace with its own Storage, Key Vault, Application Insights (required by Azure ML when the
// workspace is created) and Container Registry for the custom environments; shared user-assigned
// identity with its roles; CPU/GPU clusters that scale to zero; photo datastore; failure alert.
//
// Deployment (resource group already created):
//   az deployment group create -g parkwatch -f infra/main.bicep -p userObjectId=<id> alertEmail=<email>
// The GPU cluster needs T4 quota in Azure ML; -p deployGpuCluster=false skips it.

param location string = resourceGroup().location
param workspaceName string = 'parkwatch-workspace'
param storageName string = 'parkwatchstorage'
param keyVaultName string = 'parkwatch-keyvault'
param logAnalyticsName string = 'parkwatch-logs'
param appInsightsName string = 'parkwatch-appinsights'
param identityName string = 'parkwatch-identity'
param registryName string = 'parkwatchregistry'

@description('Storage Account with the photos (raw container).')
param dataStorageName string = 'parkwatch'
param dataStorageResourceGroup string = 'AML'

@description('Object id of the user who manages the Key Vault secrets.')
param userObjectId string

param deployGpuCluster bool = true

@description('Email that receives the failed pipeline run alerts (empty = no alert).')
param alertEmail string = ''

var tags = {
  project: 'parkwatch'
}

// Azure built-in roles.
var roles = {
  keyVaultSecretsOfficer: 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
  azureMLDataScientist: 'f6c7c914-8db3-469d-8ca1-694a8f32e121'
  acrPull: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
  storageBlobDataReader: '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'
}

// Workspace Storage (logs, snapshots, outputs). The photos stay in `parkwatch`.
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
  }
}

// RBAC: permissions are granted with roles (below), not with access policies.
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Workspace-based Application Insights (requires Log Analytics).
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// Images of the custom environments.
resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-10-01' = {
  name: workspaceName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
  properties: {
    friendlyName: workspaceName
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: registry.id
  }
}

// Identity shared by both clusters.
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
  tags: tags
}

// --- Roles on the Key Vault ---

// The workspace already has Key Vault Administrator: Azure ML assigns it on creation
// (declaring it again here clashes with RoleAssignmentExists).

// The jobs read the Label Studio token.
resource kvSecretsUserIdentity 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, identity.id, roles.keyVaultSecretsUser)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.keyVaultSecretsUser)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// The user stores and rotates the token.
resource kvSecretsOfficerUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, userObjectId, roles.keyVaultSecretsOfficer)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.keyVaultSecretsOfficer)
    principalId: userObjectId
    principalType: 'User'
  }
}

// --- Roles on the workspace and the photos ---

// The jobs register data assets and models.
resource dataScientistIdentity 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(workspace.id, identity.id, roles.azureMLDataScientist)
  scope: workspace
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.azureMLDataScientist)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// The clusters pull the environment images with their identity.
resource acrPullIdentity 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, roles.acrPull)
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.acrPull)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// check_dataset reads dataset_info.json of the latest data asset version.
resource workspaceBlobReaderIdentity 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, identity.id, roles.storageBlobDataReader)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageBlobDataReader)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// The user reads datasets and artifacts of the workspace (tools/holdout_stats.py, Studio).
resource workspaceBlobReaderUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, userObjectId, roles.storageBlobDataReader)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageBlobDataReader)
    principalId: userObjectId
    principalType: 'User'
  }
}

// Reading the photos: the `parkwatch` Storage lives in another resource group.
module photosReader 'modules/blob-reader.bicep' = {
  name: 'photos-reader'
  scope: resourceGroup(dataStorageResourceGroup)
  params: {
    storageName: dataStorageName
    principalId: identity.properties.principalId
  }
}

// --- Photo datastore (identity-based: no stored keys) ---

resource photosDatastore 'Microsoft.MachineLearningServices/workspaces/datastores@2024-10-01' = {
  parent: workspace
  name: 'parkwatch_raw'
  properties: {
    datastoreType: 'AzureBlob'
    accountName: dataStorageName
    containerName: 'raw'
    endpoint: environment().suffixes.storage
    protocol: 'https'
    credentials: {
      credentialsType: 'None'
    }
    description: 'ESP32-CAM photos (YYYY/MM/DD/parkwatch_<ts>Z.jpg)'
  }
}

// --- Clusters (0 nodes when idle) ---

var identitySettings = {
  type: 'UserAssigned'
  userAssignedIdentities: {
    '${identity.id}': {}
  }
}

resource cpuCluster 'Microsoft.MachineLearningServices/workspaces/computes@2024-10-01' = {
  parent: workspace
  name: 'cpu-cluster'
  location: location
  identity: identitySettings
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_D4as_v4'
      vmPriority: 'Dedicated'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 1
        nodeIdleTimeBeforeScaleDown: 'PT5M'
      }
      remoteLoginPortPublicAccess: 'Disabled'
    }
  }
}

resource gpuCluster 'Microsoft.MachineLearningServices/workspaces/computes@2024-10-01' = if (deployGpuCluster) {
  parent: workspace
  name: 'gpu-cluster'
  location: location
  identity: identitySettings
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_NC4as_T4_v3'
      vmPriority: 'Dedicated'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 1
        nodeIdleTimeBeforeScaleDown: 'PT5M'
      }
      remoteLoginPortPublicAccess: 'Disabled'
    }
  }
}

// --- Email alert when a pipeline run fails (block 4) ---
// The Azure ML SDK does not expose email notifications on jobs: Azure Monitor alert on the
// workspace "Failed Runs" metric, only for the pipeline experiments.

resource alertsActionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = if (!empty(alertEmail)) {
  name: 'parkwatch-alerts'
  location: 'global'
  tags: tags
  properties: {
    groupShortName: 'parkwatch'
    enabled: true
    emailReceivers: [
      {
        name: 'owner'
        emailAddress: alertEmail
        useCommonAlertSchema: true
      }
    ]
  }
}

resource failedRunsAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = if (!empty(alertEmail)) {
  name: 'parkwatch-failed-runs'
  location: 'global'
  tags: tags
  properties: {
    description: 'A run of the ParkWatch training or inference pipelines has failed.'
    severity: 2
    enabled: true
    scopes: [
      workspace.id
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    autoMitigate: true
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          criterionType: 'StaticThresholdCriterion'
          name: 'failed-runs'
          metricNamespace: 'Microsoft.MachineLearningServices/workspaces'
          metricName: 'Failed Runs'
          operator: 'GreaterThan'
          threshold: 0
          timeAggregation: 'Total'
          dimensions: [
            {
              name: 'ExperimentName'
              operator: 'Include'
              values: [
                'parkwatch-training'
                'parkwatch-inference'
              ]
            }
          ]
        }
      ]
    }
    actions: [
      {
        actionGroupId: alertsActionGroup.id
      }
    ]
  }
}

output workspaceName string = workspace.name
output keyVaultName string = keyVault.name
output identityClientId string = identity.properties.clientId
