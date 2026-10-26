# Applied Machine Learning Foundations: From Features to Real-World Decisions

This workshop is for an ODSC West session.

## Abstract

Machine learning is most useful when it helps practitioners move from data to better decisions. In this workshop, participants will learn a practical machine learning workflow: defining a prediction problem, preparing features, building tabular models, evaluating performance, and turning model scores into decisions.

Rather than focusing only on algorithm definitions, this session emphasizes the choices practitioners make when building real machine learning systems. We will cover core supervised learning concepts, feature engineering, model selection, validation strategy, and common failure modes such as target leakage, overfitting, misleading accuracy, and poorly chosen thresholds. Anomaly detection will be used as a recurring application to illustrate challenges such as limited labels, rare events, and false-positive versus false-negative tradeoffs.

Python demonstrations will use common open-source tools such as Pandas, NumPy, Matplotlib, and scikit-learn in Jupyter notebooks, with marimo, a reactive Python notebook, for select demos. The workshop will cover practical model evaluation with confusion matrices, precision, recall, F1, ROC-AUC, and PR-AUC, with attention to how metric choice depends on the cost of different errors. We will also use Flyte 2, an open-source orchestration framework, to run a scikit-learn hyperparameter search as parallel tasks that scale from a laptop to a cluster, and to turn the fraud detection workflow into a pipeline in marimo that reruns only the steps that changed.

By the end of the session, attendees will understand how to frame applied machine learning problems, prepare features for tabular models, compare model families, evaluate models beyond accuracy, scale model training with Flyte 2, and communicate model results in a way that supports real-world decisions.

![What is machine learning: supervised learning learns a known target from features; unsupervised learning finds structure in features alone](notebooks/images/machine_learning_overview.png)

## Agenda

What we will cover:

* **Framing the problem:** turn a decision into a prediction task, format data for scikit-learn, and split it so you can trust the results.
* **Core models:** linear regression, decision trees, random forests, and logistic regression, and how each one makes predictions.
* **Evaluating beyond accuracy:** rare events, leakage, precision, recall, ROC-AUC, and PR-AUC, using credit card fraud as the running example.
* **From scores to decisions:** thresholds, costs, review capacity, and calibration.
* **Scaling up:** hyperparameter search with more cores and Flyte 2, and a fraud pipeline in marimo that reruns only what changed.

<table>
  <tbody><tr>
    <th></th>
    <th>Topic</th>
    <th>Why we cover it</th>
    <th></th>
  </tr>
  <tr>
    <td>1</td>
    <td><a href="#installation"><b>Setup</b></a></td>
    <td>So everyone can run the same code during the session. Install Python and the workshop packages, including Flyte 2 and marimo.</td>
    <td></td>
  </tr>
  <tr>
    <td>2</td>
    <td><a href="./notebooks/How_to_Format_Data.ipynb"><b>How to format data for scikit-learn</b></a></td>
    <td>Every scikit-learn model expects a features matrix and a target vector. Getting this right first avoids most beginner errors.</td>
    <td><a href="./notebooks/How_to_Format_Data.ipynb"><img src="./notebooks/images/featuresMatrixTargetVector.png" width="220" alt="How to format data for scikit-learn"></a></td>
  </tr>
  <tr>
    <td>3</td>
    <td><a href="./notebooks/Linear_Regression.ipynb"><b>Linear regression</b></a></td>
    <td>The simplest model. It introduces the fit-and-predict pattern that every later model uses.</td>
    <td><a href="./notebooks/Linear_Regression.ipynb"><img src="./notebooks/images/linearregression.png" width="220" alt="Linear regression"></a></td>
  </tr>
  <tr>
    <td>4</td>
    <td><a href="./notebooks/TrainTestSplitScikitLearn.ipynb"><b>Train test split</b></a></td>
    <td>A model is only useful if it works on data it hasn't seen. This is how you check, and it shows overfitting and the bias-variance tradeoff.</td>
    <td><a href="./notebooks/TrainTestSplitScikitLearn.ipynb"><img src="./notebooks/images/TrainTestProcedure.png" width="220" alt="Train test split"></a></td>
  </tr>
  <tr>
    <td>5</td>
    <td><a href="./notebooks/Decision_Trees.ipynb"><b>Decision trees for classification</b></a></td>
    <td>An interpretable classifier. Tuning <code>max_depth</code> shows how model complexity changes results.</td>
    <td><a href="./notebooks/Decision_Trees.ipynb"><img src="./notebooks/images/decisionTreeExample.png" width="220" alt="Decision trees for classification"></a></td>
  </tr>
  <tr>
    <td>6</td>
    <td><a href="./notebooks/How_to_Visualize_Decision_Trees.ipynb"><b>Visualize decision trees</b></a></td>
    <td>Seeing the splits lets you check what the model learned and explain it to stakeholders.</td>
    <td><a href="./notebooks/How_to_Visualize_Decision_Trees.ipynb"><img src="./notebooks/images/plottreefncn.png" width="220" alt="Visualize decision trees"></a></td>
  </tr>
  <tr>
    <td>7</td>
    <td><a href="./notebooks/RandomForestUsingPython.ipynb"><b>Random forests</b></a></td>
    <td>Many trees together usually beat one tree on tabular data. Also covers feature importance and grid search.</td>
    <td><a href="./notebooks/RandomForestUsingPython.ipynb"><img src="./notebooks/images/Bagging.png" width="220" alt="Random forests"></a></td>
  </tr>
  <tr>
    <td>8</td>
    <td><a href="./notebooks/Logistic_Regression.ipynb"><b>Logistic regression</b></a><br><sub><a href="./notebooks/Logistic_Regression_for_Multiclass.ipynb">Multiclass</a> · <a href="https://medium.com/@GalarnykMichael/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1">Speeding up training</a></sub></td>
    <td>A classifier that outputs probabilities, which the fraud notebooks later turn into decisions. The multiclass notebook handles more than two classes.</td>
    <td><a href="./notebooks/Logistic_Regression.ipynb"><img src="./notebooks/images/logistic_regression_predictions.png" width="220" alt="Logistic regression"></a></td>
  </tr>
  <tr>
    <td>9</td>
    <td><a href="./notebooks/Anomaly_Detection_Evaluation.ipynb"><b>Anomaly detection: evaluating models beyond accuracy</b></a><br><sub>Downloads the <a href="#dataset">dataset</a> on first run</sub></td>
    <td>Fraud is rare, so a model that never flags anything is 99.8% accurate. Covers leakage, comparing model families, and metrics like PR-AUC that show real differences.</td>
    <td><a href="./notebooks/Anomaly_Detection_Evaluation.ipynb"><img src="./notebooks/images/fraud_confusion_matrix.png" width="220" alt="Anomaly detection: evaluating models beyond accuracy"></a></td>
  </tr>
  <tr>
    <td>10</td>
    <td><a href="./notebooks/Turning_Scores_into_Decisions.ipynb"><b>Turning model scores into decisions</b></a></td>
    <td>A model gives a score, but the business needs an action. Covers choosing a threshold from costs, review capacity, and calibration.</td>
    <td><a href="./notebooks/Turning_Scores_into_Decisions.ipynb"><img src="./notebooks/images/cost_vs_threshold.png" width="220" alt="Turning model scores into decisions"></a></td>
  </tr>
  <tr>
    <td>11</td>
    <td><a href="./notebooks/Hyperparameter_Tuning.ipynb"><b>Does additional compute always help? Hyperparameter tuning with more cores and Flyte 2</b></a><br><sub><a href="#optional-run-flyte-on-a-local-cluster">Optional cluster run</a></sub></td>
    <td>Tuning is where most compute time goes. On the full fraud dataset, a grid search drops from about 30 minutes on one core to under 3 minutes on 16.</td>
    <td><a href="./notebooks/Hyperparameter_Tuning.ipynb"><img src="./notebooks/images/hyperparameter_timing_full_dataset.png" width="220" alt="Does additional compute always help? Hyperparameter tuning with more cores and Flyte 2"></a></td>
  </tr>
  <tr>
    <td>12</td>
    <td><a href="./notebooks/Flyte_Pipeline_with_marimo.py"><b>The fraud detection pipeline with Flyte 2 and marimo</b></a><br><sub><a href="#running-the-marimo-notebook">How to run it</a></sub></td>
    <td>Puts the whole fraud workflow into one pipeline. When you change one step, only that step reruns.</td>
    <td></td>
  </tr>
</tbody></table>

## Installation

The packages (scikit-learn, Jupyter, Flyte 2, marimo, and the rest) are listed in `requirements.txt` for uv and in `environment.yml` for conda. Use either option below. Docker is not needed for the notebooks. It is only used for the [optional cluster runs](#optional-run-flyte-on-a-local-cluster) and is installed separately.

### Option 1: uv (recommended)

1. Install uv:

   ```
   # Mac and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Open a new terminal so the `uv` command is found.

2. From the repo folder, create the environment and install the packages. uv downloads Python 3.11 if you don't have it.

   ```
   uv venv --python 3.11
   uv pip install -r requirements.txt
   ```

3. Get into the environment and start JupyterLab:

   ```
   # Mac and Linux
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate

   jupyter lab
   ```

   Run the activate command again from the repo folder every time you open a new terminal. Leave the environment with `deactivate`.

### Option 2: conda

1. Install conda if `conda --version` gives "command not found". [Miniforge](https://github.com/conda-forge/miniforge) is a small conda installer (on Windows, download the installer from that page instead):

   ```
   # Mac (Apple Silicon)
   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
   bash Miniforge3-MacOSX-arm64.sh -b
   ~/miniforge3/bin/conda init zsh
   ```

   Open a new terminal so the `conda` command is found. Anaconda also works if you already have it.

2. From the repo folder, create the environment:

   ```
   conda env create -f environment.yml
   ```

3. Get into the environment and start JupyterLab:

   ```
   conda activate sklearn-workshop
   jupyter lab
   ```

   Run `conda activate sklearn-workshop` every time you open a new terminal. Leave the environment with `conda deactivate`.

### Option 3: Google Colab (nothing to install)

If you can't install software on your computer, open a notebook in [Google Colab](https://colab.research.google.com/) from GitHub (File > Open notebook > GitHub, then paste this repo's URL). scikit-learn, pandas, and matplotlib are already installed. Run this in the first cell so the notebook can find its data files and the Flyte packages:

```
!git clone https://github.com/mGalarnyk/MLModelsScikit-Learn.git
%cd MLModelsScikit-Learn/notebooks
!pip install flyte pyarrow
```

The marimo notebook (`Flyte_Pipeline_with_marimo.py`) and the optional cluster runs need a local install (Option 1 or 2).


## Running the Notebooks

### Dataset

The fraud detection notebooks use the real [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, 492 frauds). They download it (about 150 MB) to `notebooks/data/creditcard.csv` the first time they run, so the first run needs an internet connection.

### Running the marimo notebook

`Flyte_Pipeline_with_marimo.py` is a marimo notebook, not a Jupyter notebook. Start it from the `notebooks` folder:

```
cd notebooks
marimo edit Flyte_Pipeline_with_marimo.py   # notebook view, with code
marimo run Flyte_Pipeline_with_marimo.py    # app view, code hidden
```

A browser tab opens. Choose the models, leave "Local" selected, and click "Run pipeline with Flyte". The first run takes about a minute. Then add Gradient-Boosted Trees and run again: the pipeline diagram shows which Flyte tasks were reused from the cache and which ran. To start fresh, delete the cache with `rm -rf ~/.flyte/local-cache`.

### Optional: Run Flyte on a Local Cluster

Flyte 2 runs locally without Docker, so every notebook works without it. Docker is only needed to run the same code on a local Flyte cluster:

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it. Docker is not in `requirements.txt` or `environment.yml`, so install it separately.
2. Start the cluster: `flyte start devbox --apple-silicon` (drop `--apple-silicon` on other machines). The Flyte UI is at http://localhost:30080/v2.
3. In the `notebooks` folder, run `flyte create config --devbox`, then set `RUN_ON_CLUSTER = True` in `Hyperparameter_Tuning.ipynb` or pick "Flyte devbox" in the marimo notebook.
4. Stop the cluster afterwards with `flyte stop devbox`.

## Relevant Links

These resources were shared during audience Q&A.

### Machine Learning
- [Understanding Sampling With and Without Replacement (Python)](https://towardsdatascience.com/understanding-sampling-with-and-without-replacement-python-7aff8f47ebe4/)  
  Explains random sampling principles using clear Python examples.
- [Understanding Random Forest using Python (scikit-learn)](https://youtu.be/R9tJeEgHyeo?si=xKVoLu5eb7v2CgcY)  
  Video walkthrough of how ensemble tree models combine multiple learners.
- [Understanding Decision Trees for Classification (Python)](https://medium.com/data-science/understanding-decision-trees-for-classification-python-9663d683c952)  
  Step-by-step explanation of decision tree construction, splitting, and interpretation.

### Curriculum Learning
- [On the Creation of Narrow AI: Hierarchy and Nonlocality of Neural Network Skills](https://arxiv.org/pdf/2505.15811)  
  Discusses how neural networks develop hierarchical skills and how this relates to curriculum learning and skill transfer.

### Model Deployment and Maintenance
- [Considerations for Deploying Machine Learning Models in Production](https://medium.com/data-science/considerations-for-deploying-machine-learning-models-in-production-89d38d96cc23)  
  Covers practical issues like reproducibility, model retraining, and monitoring.
- [What is Data Drift in ML, and How to Detect and Handle It](https://medium.com/data-science/considerations-for-deploying-machine-learning-models-in-production-89d38d96cc23)  
  Explains how changing data distributions can degrade model performance and what to do about it.

### Reinforcement Learning
- [An Introduction to Reinforcement Learning with OpenAI Gym, RLlib, and Google Colab](https://www.anyscale.com/blog/an-introduction-to-reinforcement-learning-with-openai-gym-rllib-and-google)  
  Practical overview of RL concepts and tools with runnable examples.
- [REFT: Reasoning with REinforced Fine-Tuning](https://arxiv.org/pdf/2401.08967)  
  Paper exploring reinforcement learning methods for reasoning and model fine-tuning.

