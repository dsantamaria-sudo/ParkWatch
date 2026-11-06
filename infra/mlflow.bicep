// ParkWatch — self-hosted MLflow tracking server (experiment comparison).
// Container App that scales to zero, image mlflow_server/ in parkwatchregistry, tracking store in
// the PostgreSQL server `mlflowtest` (databases parkwatch_mlflow + parkwatch_mlflow_auth, role
// `mlflow`), artifacts in the `mlflow` container of the workspace Storage, written with its own
// identity. Secrets are Key Vault references; nothing secret is passed as a parameter.
// The model registry and the champion stay in Azure ML: this server is for tracking only.
//
// Prerequisites: infra/main.bicep deployed, image pushed (az acr build), Key Vault secrets
// `postgres-mlflow-password`, `mlflow-admin-password` and `mlflow-flask-secret-key` (CSRF key of the
// login forms; it must stay the same across restarts), databases created with
// mlflow_server/create_databases.py.
//
// Deployment:
//   az deployment group create -g parkwatch -f infra/mlflow.bicep

param location string = resourceGroup().location
param storageName string = 'parkwatchstorage'
param keyVaultName string = 'parkwatch-keyvault'
param logAnalyticsName string = 'parkwatch-logs'
param registryName string = 'parkwatchregistry'

param identityName string = 'parkwatch-mlflow-identity'
param environmentName string = 'parkwatch-apps'
param appName string = 'parkwatch-mlflow'
param image string = 'mlflow-server:3.18.0-4'

@description('Migrate the tracking schema on start: only after changing the MLflow version.')
param mlflowDbUpgrade bool = false

param postgresHost string = 'mlflowtest.postgres.database.azure.com'
param postgresUser string = 'mlflow'
param trackingDatabase string = 'parkwatch_mlflow'
param authDatabase string = 'parkwatch_mlflow_auth'
param artifactsContainer string = 'mlflow'

var tags = {
  project: 'parkwatch'
}

var roles = {
  acrPull: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
  storageBlobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageName
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: logAnalyticsName
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: registryName
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' existing = {
  parent: storage
  name: 'default'
}

resource artifacts 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: artifactsContainer
  properties: {
    publicAccess: 'None'
  }
}

// Own identity: the server only gets what it needs (separate from the pipeline identity).
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
  tags: tags
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, roles.acrPull)
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.acrPull)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource kvSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, identity.id, roles.keyVaultSecretsUser)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.keyVaultSecretsUser)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Write access limited to the artifacts container.
resource artifactsWriter 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(artifacts.id, identity.id, roles.storageBlobDataContributor)
  scope: artifacts
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageBlobDataContributor)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource appsEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: appsEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 5000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: registry.properties.loginServer
          identity: identity.id
        }
      ]
      secrets: [
        {
          name: 'pg-password'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/postgres-mlflow-password'
          identity: identity.id
        }
        {
          name: 'admin-password'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/mlflow-admin-password'
          identity: identity.id
        }
        {
          name: 'flask-secret-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/mlflow-flask-secret-key'
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'mlflow'
          image: '${registry.properties.loginServer}/${image}'
          // 1 CPU / 2 GiB was OOM-killed under UI load (parent + 2 workers of MLflow 3.x), and every
          // kill meant minutes of crash-loop back-off. 2 CPU / 4 GiB is the next valid combination.
          resources: {
            cpu: json('2.0')
            memory: '4Gi'
          }
          // MLflow 3.x needs ~40 s to migrate and import before it listens on 5000: the default
          // startup probe killed it in a loop. /health is exempt from auth and host validation.
          probes: [
            {
              type: 'Startup'
              tcpSocket: { port: 5000 }
              initialDelaySeconds: 10
              periodSeconds: 5
              timeoutSeconds: 3
              failureThreshold: 36
            }
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 5000 }
              periodSeconds: 30
              timeoutSeconds: 10
              failureThreshold: 3
            }
          ]
          env: [
            { name: 'PG_HOST', value: postgresHost }
            { name: 'PG_USER', value: postgresUser }
            { name: 'PG_PASSWORD', secretRef: 'pg-password' }
            { name: 'PG_DB', value: trackingDatabase }
            { name: 'PG_AUTH_DB', value: authDatabase }
            { name: 'MLFLOW_AUTH_ADMIN_PASSWORD', secretRef: 'admin-password' }
            { name: 'MLFLOW_FLASK_SERVER_SECRET_KEY', secretRef: 'flask-secret-key' }
            { name: 'ARTIFACTS_DESTINATION', value: 'wasbs://${artifactsContainer}@${storageName}.blob.${environment().suffixes.storage}/' }
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
            // Two workers: a slow UI request does not block the rest (nor the liveness probe).
            { name: 'MLFLOW_WORKERS', value: '2' }
            // MLflow 3.x starts a job runner by default (one more process with MLflow loaded, plus
            // a queue in /dev/shm) for GenAI features this server does not use.
            { name: 'MLFLOW_SERVER_ENABLE_JOB_EXECUTION', value: 'false' }
            // 'true' only for the first deployment after an MLflow upgrade (schema migration).
            { name: 'MLFLOW_DB_UPGRADE', value: mlflowDbUpgrade ? 'true' : 'false' }
            // MLflow 3.x rejects requests whose Host header is not listed (DNS rebinding protection).
            { name: 'ALLOWED_HOSTS', value: '${appName}.${appsEnv.properties.defaultDomain},localhost,localhost:*' }
            { name: 'CORS_ALLOWED_ORIGINS', value: 'https://${appName}.${appsEnv.properties.defaultDomain}' }
          ]
        }
      ]
      // Scale to zero: the first request after idle takes ~30 s (cold start).
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
  dependsOn: [
    acrPull
    kvSecretsUser
    artifactsWriter
  ]
}

output mlflowUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
