# A Repository With Code Connected to the UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN) Kaggle Challenge 
Information about the challenge can be found [here](https://www.kaggle.com/competitions/UBC-OCEAN).

# Guide to Weights & Biases

## Sign up

Create a wandb account: go to https://wandb.ai/
Request an invite to our wandb team: Message your email to Elias
Find your api key at https://wandb.ai/home
Create a .env file in the root directory and add the following line: `WANDB_API_KEY=[your api key]`

## Sign in

### Import api key from .env

`from dotenv import dotenv_values`
`env_config = dotenv_values(".env")`

### Sign into wandb

`import wandb`
`wandb.login(key=env_config["WANDB_API_KEY"])`

### Using the init_experiment() function

1. Create the config dictionary with all of the meta data and hyperparameters (needs to have an architecture).
2. Call `init_experiment(wandb, project_name, experiment_name, extra, config)`
3. Use `wandb.log({"metric": value})` to log metrics.
4. View results in /wandb or on the wandb website.
