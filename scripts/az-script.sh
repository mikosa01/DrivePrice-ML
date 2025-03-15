

az ad sp create-for-rbac --name "github-actions" --role "Contributor" \
    --scopes $(az acr show --name DriveML --query id --output tsv) \
    --sdk-auth

az containerapp env  create --name "driveml-env" \
                    --resource-group "car_sales" \
                    --location "UK South"