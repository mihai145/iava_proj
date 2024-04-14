# Used to test the custom solver locally.

from generic_dataset import GenericDataset, VisualizationDataset, CELEBA_FORMAT_DATASET, RAFD_FORMAT_DATASET, get_loader
from custom_solver import CustomSolver


# Train Datasets
celeba = GenericDataset("data/celeba", CELEBA_FORMAT_DATASET, ["Male", "Young", "Arched_Eyebrows", "Black_Hair", "Blond_Hair", "Brown_Hair"], "train")
rafd = GenericDataset("data/rafd", RAFD_FORMAT_DATASET, ["happy", "moustache", "sad"], "train")
# third = GenericDataset("data/third", RAFD_FORMAT_DATASET, ["bald", "angry"], "train")
# fourth = GenericDataset("data/fourth", RAFD_FORMAT_DATASET, ["helmet"], "train")

# Visualization Dataset
visualization_ds = VisualizationDataset("data/vis_src")

# Override defaults to test locally
override_defaults = {
    "num_iters": 20,
    "n_critic": 5,
    "log_step": 5,
    "vis_step": 5,
    "model_save_step": 5,
    "num_iters_decay": 10,
    "lr_update_step": 2,
}

# solver = CustomSolver([celeba, rafd, third, fourth], visualization_ds, "experiment_1", **override_defaults)
solver = CustomSolver([celeba, rafd], visualization_ds, "experiment_2", **override_defaults)
solver.train_multi()
