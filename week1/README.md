# Week 1 content

## Introduction to git
Git is a version control system that allows you to track changes in your code and collaborate with others.

Basic commands that you should know (either in the command line or in a GUI):
- `git clone <repo>`: Clone a repository from GitHub to your local machine.
- `git commit -m "<message>"`: Commit your changes with a message.
- `git push`: Push your changes to the remote repository.
- `git pull`: Pull the latest changes from the remote repository.

What you should do:
- Create a GitHub account if you don't have one.
- Create a new repository on GitHub (maybe create a branch of this one).
- Clone the repository to your local machine.
- Practice the basic commands listed above.

## Introduction to Miniconda
Miniconda is a minimal installer for conda, a package manager that helps you manage your Python environments and packages.

Basic commands that you should know:
- `conda create -n <env_name> python==<version>`: Create a new conda environment with a specific Python version.
- `conda activate <env_name>`: Activate the conda environment.
- `conda deactivate`: Deactivate the current conda environment.
- `conda install <package>` or `pip install <package>`: Install a package in the current conda environment.
- `conda export env > environment.yml`: Export the current conda environment to a YAML file.
- `conda env create -f environment.yml -n <env_name>`: Create a new conda environment from a YAML file.
- `pip freeze > requirements.txt`: Export the current Python packages to a requirements file.
- `pip install -r requirements.txt`: Install the packages listed in a requirements file.

What you should do:
- Install Miniconda on your machine.
- Create a new conda environment with Python 3.9 or higher.
- Install the packages listed in the `week1.yml` file.
- Add any additional packages you want to use in your project.
- Export your conda environment to a YAML file.

## Introduction to Jupyter Notebooks
Jupyter Notebooks are interactive documents that allow you to write and run code, visualize data, and create reports. They are widely used in data science and machine learning.

How to run a Jupyter Notebook:
- Open it through an integrated development environment (IDE) like VSCode or PyCharm.
- Make sure you have `ipykernel` installed in your conda environment.
- Click the "Run" button or use the keyboard shortcut (Ctrl + Enter) to run a cell.
- Use Markdown cells to write text, explanations, and comments.
- Use code cells to write and run Python code.

## Introduction to AI/ML
Refer to 'week1/why.ipynb' for a brief introduction to AI/ML. Think of more use cases of why a certain problem needs to be solved using AI/ML, rather than traditional programming.
