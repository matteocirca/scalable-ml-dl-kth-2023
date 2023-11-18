# Task 1 - Iris Flower classifier

The following is the sequence of operations performed to deploy task's 1 final pipeline .
## 1.1 Feature Pipeline
### 1.1.1 Dataset Analysis and Feature Group Creation

By executing the notebook `iris-eda-and-backfill-feature-group.ipynb`, the Iris flower dataset is downloaded and analysis is performed, such as composition, distribution of values and correlation between features.
Finally, a [Feature Group](https://docs.hopsworks.ai/feature-store-api/2.5.9/generated/feature_group/) is created, with [Expectations](https://docs.hopsworks.ai/feature-store-api/2.5.9/generated/api/expectation_api/) in order to validate the data, and uploaded to Hopsworks.

### 1.1.2 Creating synthetic flowers

To simulate the periodic arrival of new entries in the dataset, the script `iris-feature-pipeline-daily.py` creates and adds to the Feature Group created in Section 1.1.1 new flowers generated using random values that respect the boundaries of values possible for generating a certain type of flower. This script is then uploaded on [Modal](https://modal.com/) and run daily. 

## 1.2 Training Pipeline
### 1.2.1 Training, Testing and Registering the model

By executing the notebook `iris-training-pipeline.ipynb`, the feature group created in the previous step is downloaded and used in order to train a [KNN classifier](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). Once the latter is trained, performance metrics as well as the confusion matrix of the prediction are plotted. Finally, the model is exported and uploaded on Hopsworks's [Model Registry](https://docs.hopsworks.ai/3.1/concepts/mlops/registry/).


## 1.3 Inference Pipeline
### 1.3.1 Evaluating new flowers

To assess the model's performances with the new synthetic flowers inserted daily, the script `iris-batch-inference-pipeline.py` downloads the model from Hopsworks's Model Registry and runs a prediction on the whole dataset. After gathering the predictions, the script appends the last prediction to the history of predictions of the syntethic data, saving both a picture of the history of predictions and the confusion matrix (only if one prediction for class of flower took place) in Hopsworks FS.
This script, as for the one in Section 1.1.2 is deployed on Modal in order to run it daily.

### 1.3.2 Deploying on Hugging Face

To access the infrastructure that we deployed in the previous sections, [spaces](https://huggingface.co/docs/hub/spaces-overview) were deployed on HuggingFace in order to run the model created. 
Two spaces were created: [iris](https://matteocirca-iris.hf.space/) and [iris-monitor](https://matteocirca-iris-monitor.hf.space/).
The first acts as an interface where the user can input values for each of the features and get a prediction on which type of flower reflects the values inserted, the latter shows the monitoring features created in Section 1.3.2.



