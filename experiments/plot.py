from src.utils.experiment_utils import create_plots




models = [
    "exp1/",
    "exp2/",
    "exp3/"
]


for model_name in models:

    create_plots(f"artifacts/{model_name}",f"report/figures/{model_name}")
