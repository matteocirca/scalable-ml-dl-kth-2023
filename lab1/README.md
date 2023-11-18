# Task 1 - Iris Flower classifier

The following is the infrastructure built to deploy task 1's pipeline.
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
The first acts as an interface where the user can input values for each of the features and get a prediction on which type of flower reflects the values inserted, the latter shows the monitoring for inference on flowers created in Section 1.1.2.

# Task 2 - Wine Quality classifier

The following is the infrastructure built to deploy task 2's pipeline.
## 2.1 Feature Pipeline
### 2.1.1 Dataset Analysis and Feature Group Creation

By executing the notebook `wine-eda-and-backfill-feature-group.ipynb`, the wine quality dataset is downloaded and analysis is performed, such as composition, distribution of values, and correlation between features.
The dataset presents some peculiarity that we needed to address before proceeding: 
-  NaNs: Some values presented NaN values for some features. For simplicity, we decided to remove the 38 rows presenting this defect
-  Duplicated rows: The dataset presented 1168 duplicated rows, since this can affect the training quality we dropped them.
-  Quality: the label of the dataset seems to be meant to have values in the range [0,10], but there is a strong imbalance between the present values, with some values not making it into the dataset at all or with very few samples. To address this problem, we decided to binarize the labels into "Good" quality (quality $\ge$ 6) and "Bad quality" (quality $\lt$ 6).
-  Wine Type: the dataset presents a categorical feature to express the type of wine ("white" or "red"). We substituted the two values with a 0-1 value.

Finally, a [Feature Group](https://docs.hopsworks.ai/feature-store-api/2.5.9/generated/feature_group/) is created, with [Expectations](https://docs.hopsworks.ai/feature-store-api/2.5.9/generated/api/expectation_api/) in order to validate the data, and uploaded to Hopsworks.

### 2.1.2 Creating synthetic wines

To simulate the periodic arrival of new entries in the dataset, the script `wine-feature-pipeline-daily.py` creates and adds to the Feature Group created in Section 2.1.1 new wines generated using random values.
Since the number of samples per features is high enough, it is possible to apply the [Central Limit Theorem](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5370305/) and assume that every feature's distribution can be approximated to a distribution $\mathcal{X}\sim\mathcal{N}(\mu,\sigma^2)$, with $\mu$ the mean and $\sigma^2$ the standard deviation of the feature's values. Because different wine types can highly impact the values of the features, the upper-cited method is used taking into account the type of wine (computing different means and standard deviations based on the type).
Differently from the script used for the iris flower generation, it is not possible to know in advance the label of the wine generated. To address this problem we run an inference task on the generated features and use the predictions as labels for the new synthetic wines. 
This script is finally uploaded on [Modal](https://modal.com/) and runs daily. 

## 2.2 Training Pipeline
### 2.2.1 Training, Testing and Registering the model

By executing the notebook `wine-training-pipeline.ipynb`, the feature group created in the previous step is downloaded and used in order to train a [KNN classifier](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). However, [a study on the same dataset](https://www.kaggle.com/code/wumanandpat/wine-quality-finding-the-minimal-model) demonstrated that the minimal error in the model is obtained using a [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) Classifier, so we decided to perform the classification tasks using the latter model.

Once the model is trained, performance metrics as well as the confusion matrix of the prediction are plotted. Finally, the model and performance metrics are exported and uploaded on Hopsworks's [Model Registry](https://docs.hopsworks.ai/3.1/concepts/mlops/registry/).


## 2.3 Inference Pipeline
### 2.3.1 Evaluating new wines

To assess the model's performances with the new synthetic wines inserted daily, the script `wine-batch-inference-pipeline.py` downloads the model from Hopsworks's Model Registry and runs a prediction on the whole dataset. After gathering the predictions, the script appends the last prediction to the history of predictions of the synthetic data, saving both a picture of the history of predictions and the confusion matrix (only if one prediction for the class of wine took place) in Hopsworks FS.
This script, as for the one in Section 2.1.2 is deployed on Modal in order to run it daily.

### 2.3.2 Deploying on Hugging Face

To access the infrastructure that we deployed in the previous sections, [spaces](https://huggingface.co/docs/hub/spaces-overview) were deployed on HuggingFace in order to run the model created. 
Two spaces were created: [wine](https://matteocirca-wine.hf.space/) and [wine-monitor](https://matteocirca-wine-monitor.hf.space/).
The first acts as an interface where the user can input values for each of the features and get a prediction on whether those features reflect a good or bad wine, the latter shows the monitoring for inference on wines created in Section 2.1.2.



