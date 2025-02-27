## Guide to deploy AIRP

### HOLTER lIBRARY

- branch:
  - `ai_deploy`: Using to services using Tensorflow Library, to run AI model.
  - `ai_deploy_without_tensorflow`: Using to services not using Tensorflow Library.

- Increase version of `btcy_holter`
  - Get the current version in `Makefile`: `PYTHON_PACKAGE_VERSION=x.x.x` or `btcy_holter/version.py`
  - Replace the current version to the new version: `x.x.x` => `x.x.{x+1}`
    - `Makefile`
    - `btcy_holter/version.py`
  - Deploy lib by

    ```shell
    make deploy-lib
    ```

- Build docker base image:
  - Get the current version in `Makefile`: `ECR_BASE_TF_TAG=x.x.x`
  - Replace the current version to the new version: `x.x.x` => `x.x.{x+1}`
  - Deploy lib by

    ```shell
    make docker-base-build
    make docker-base-push
    ```

### AI-HOURLY-ANALYZER

- **Note**: using btcy_holter version at branch `ai_deploy`

- Get the current version in `Makefile`: `DEV_TAG=x.x.x`

- Replace the current version to the new version: `x.x.x` => `x.x.{x+1}`
  - Makefile
  - deloy
    - chart.yaml: `version: x.x.{x+1}`
    - values.yaml: `tag: "x.x.{x+1}"`
  - deloy-heavy
    - chart.yaml: `version: x.x.{x+1}`
    - values.yaml: `tag: "x.x.{x+1}"`

- Check the configuration of the TFServer environments in `values.yaml`

    ```
    # customer
    # value: tf-hourly-analyzer-service

    # staging
    value: tf-event-analyzer-service

    # delta
    # value: demo-tf-server-service.alpha.svc

    # alpha
    # value: demo-tf-server-service
    ```

- If has update the btcy_holter library
  - Change the version `btcy_holter` in `Makefile`: `PY_PACKAGE_VERSION={version of btcy_holter}`
  - Change the version of docker image in `Dockerfile`: `FROM 223480282509.dkr.ecr.us-east-2.amazonaws.com/base-ai-gateway:{version of docker base image}`

- Git commit:

```
Version x.x.{x+1}: {description}
```

### AI-HOLTER-PROCESSOR

- **Note**: using btcy_holter version at branch `ai_deploy_without_tensorflow`

- Get the current version in `Makefile`: `DEV_TAG=x.x.x`

- Replace the current version to the new version: `x.x.x` => `x.x.{x+1}`
  - Makefile
  - deloy-flex
    - chart.yaml: `version: x.x.{x+1}`
    - values.yaml: `tag: "x.x.{x+1}"`
  - deloy-stable
    - chart.yaml: `version: x.x.{x+1}`
    - values.yaml: `tag: "x.x.{x+1}"`

- If has update the btcy_holter library
  - Change the version `btcy_holter` in `Makefile`: `PY_PACKAGE_VERSION={version of btcy_holter}`
  - Change the version of docker image in `Dockerfile`: `FROM 223480282509.dkr.ecr.us-east-2.amazonaws.com/base-ai-gateway:{version of docker base image}`

- Git commit:

```
Version x.x.{x+1}: {description}
```

## Monitor the progress of the build

    - alpha/delta: [Link](https://deploy.bioflux.io/view/Bioflux%20AI/)
    - staging/customer: [Link](https://staging.ops.biotricity.com/view/Bioflux%20-%20AI/)