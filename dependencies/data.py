import os
from dependencies.workspace import Workspace

# Set location of local data directory from environment variable
env_var = 'DATA'
if env_var not in os.environ:
    assert False, env_var + ' environmental variable must be set.'
root = os.environ[env_var]

# define workspace
data_workspace = Workspace(root)
workspace = data_workspace  # for reference in third_party textmorph implementation.

# Preprocessing runs
data_workspace.add_dir('datasets', 'datasets')

# Training runs
data_workspace.add_dir('edit_runs', 'training_runs')

# Logs App level
data_workspace.add_dir('logs', 'logs')

# define the code workspace
code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
code_workspace = Workspace(code_root)
code_workspace.add_dir("configs", "configs")
