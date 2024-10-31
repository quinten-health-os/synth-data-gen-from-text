# synth-data-gen-from-text

## Overview

This project proposes a framework for generation of synthetic tabular data from text using large language models (LLMs).
We propose an approach to generate synthetic tabular patient data that does not require access to the original data, but only a description of the desired database. We leverage prior medical knowledge and in-context learning capabilities of LLMs to generate realistic patient data, even in a low-resource setting. We quantitatively evaluate our approach against state-of-the-art SDG models, using fidelity, privacy, and utility metrics.
This approach, which is easy to use and does not require original data or advanced ML skills, is particularly valuable for quickly generating custom-designed patient data, supporting project implementation and providing educational resources.

The project was developed using data from the Parkinson’s Progression Markers Initiative (PPMI)(RRID:SCR 006431) and Alzheimer's Disease Neuroimaging Initiative(ADNI) database. For up-to-date information on the studies, visit www.ppmi-info.org and https://adni.loni.usc.edu.

For more information on the developped framework, see our paper [here].

## Installation

### Requirements

- Python ~3.10

### Environment set-up

To set-up the project for the first time run the following commands:

```bash
git clone
cd synth-data-gen-from-text && make build-docker-dev-image && make create-docker-dev-container
```

Then

1. Run `make start-docker-dev-container` to start the container associated with the project.

2. Use [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) to attach the running container (Command Palette > `Dev Containers: Attach to Running Container`) to VSCode. 

### Addition of new package

```bash
poetry add <package_name>==<package_version> # install the dependency, add it to pyproject.toml and regenerate the poetry.lock file
```

Have look a [here](https://python-poetry.org/docs/cli/#add) for more details.

### Data 

* Request access to PPMI data from the PPMI Data Access Committee  [here](https://www.ppmi-info.org/access-data-specimens/download-data)
* Request acess to the ADNI data from LONI platform [here](https://ida.loni.usc.edu/collaboration/access/appLicense.jsp;jsessionid=2DC933ADA7B32AFD298B0C4BF6DF54AC)



## Usage
#### Data storage

The framework was developped using an AWS s3 storage.
To adapt the project to your s3 storage you can change the bucket name in the `config.py file`: `BUCKET_NAME`.

In case you want to work with another storage system, you can also adapt the loading and saving function of input and output tables in the script `src/loading.py`.

#### Description of pipeline

The pipeline include the following steps each corresponding to a script included in the `./scripts` folder:

1. Data preparation
- preparing


2. Synthetic data generation
- `text_to_tab_sdg`: SDG using a prompt and a LLM
- `text_to_tab_sdg_shuffle`
- `tab_to_tab_sdg` : SDG using original data and state-of-the-art SDG models


3. Evaluation
- `evaluate_fidelity`
- `evaluate_privacy`
- `evaluate_utility`
- `evaluate`
- `evaluate_agg`
- `describe_data`

Each script necessitates parameters that are specified in the `config.py` file.

To reproduce the results of our paper for our approach (GPT-4), here is how to set key parameters to run the pipeline steps:

```
PIPELINE_STEPS_TO_PERFORM = [
    "preparing",
    "text_to_tab_sdg_shuffle",
    "describe_data",
    "evaluate",
    "evaluate_agg"
]
DATABASE = "ppmi2024"
SDG_MODEL = "gpt-4-turbo" 
RANDOM_STATE = 1
N_ROWS = 10
N_SAMPLE = 1000
PROMPT_ID = "ppmi_prompt" 
```

and run 
``` bash
python scripts/main_pipeline_train_test.py
```
We additionnaly provide the possibility to run the pipeline steps with mlflow (to configure in the `config.py` file) with :
``` bash
python scripts/main_pipeline_mlflow.py
```
To run the pipeline for any SDG tabular-to-tabular model, you can replace the step `text_to_tab_sdg_shuffle` by `tab_to_tab_sdg` and chose `SDG_MODEL` among `copula`, `tvae`, `ctgan` and set `PROMPT_ID` to the name of the database e.g. `ppmi` or `adni`.

To run the pipeline without the evaluation framework, you can simply remove `evaluate_agg` from the pipeline steps and run:

``` bash
python scripts/main_pipeline.py
```

## Support

You can contact the repository's maintainers if support is needed.

## Contributing

Want to contribute? No problem, there are two ways to contribute:

- Open issues on the project's Gitlab.
- Resolve issues by creating new branches and merge requests, which will be reviewed before being merged into the repository.







