# Databricks notebook source
# MAGIC %md
# MAGIC # Context
# MAGIC Our approach is to exploit `ResNet-50`, a convolutional neural network of 50 layers deep that was trained over more than a million images from the ImageNet database. By removing `ResNet-50` top layer, we can build our own model that can detect accident severity. Someone has already spent the time and computed resources to learn a lot of features and our model will likely benefit from it. This approach is known as **Transfer Learning**. We report below a complex deep learning architecture made of several convolutional layers and rectified linear unit (ReLU) activation functions. We can easily "peel off" its last layer (classification) to turn this model into an efficient image featurizer without the computational burden of training a model on our own. This approach will drastically reduce the time from ideation to business value.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png" width="600">
# MAGIC <br>
# MAGIC 
# MAGIC [neurohive.io](neurohive.io)

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC # Transfer learning
# MAGIC In our previous notebook, we showed how to create a collection of images displaying accidents (labelled 1) or not (labelled 0) as a simple delta table. The aim will be not to train a classifier, but a regression model to detect the likelihood of a car being damaged or not, 0 being an undamaged car and 1 being a wreck. In this first section, we will convert raw images into feature vectors.

# COMMAND ----------

from pyspark.sql import functions as F
input_df = spark.read.table(config['database']['tables']['images']).withColumn('label', F.col('label').cast('DOUBLE'))
display(input_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading model
# MAGIC We load our pre-trained neural network and drop its last layer (classification) since we will build our own regression model. In a distributed environment, we ensure each executor has access to our pre-train pipeline by broadcasting its model weights

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# load ResNet50 model with top layer peeled off
# we only use model CNN for image featurization and will use our own classification
# this approach is called transfer learning
model = ResNet50(include_top = False)

# broadcast model layers (weights) to executors so that each executor can load model faster
bc_model_weights = sc.broadcast(model.get_weights())

# COMMAND ----------

import pandas as pd
import numpy as np
import io

def load_model():
  '''
  :return: a ResNet50 model with top layer removed and broadcasted pretrained weights.
  Each executor will be able to quickly load a pre-train model to perform image featurization
  Again, we are not training a new CNN but simply use a pre-trained model to transform 
  raw images into input vectors for our classification model
  '''
  from tensorflow.keras.applications.resnet50 import ResNet50
  model = ResNet50(weights=None, include_top=False)
  model.set_weights(bc_model_weights.value)
  return model

def preprocess(content):
  '''
  Preprocesses raw image bytes for prediction.
  open image from binary data
  resize picture to 224x224
  convert image to array using Keras pre-processing img_to_array function
  pre-process input array through Keras Resnet50 functionality
  '''
  from PIL import Image
  from tensorflow.keras.applications.resnet50 import preprocess_input
  from tensorflow.keras.preprocessing.image import img_to_array
  img = Image.open(io.BytesIO(content))
  arr = img_to_array(img)
  model_input = preprocess_input(arr)
  return model_input

def featurize_series(model, content_series):
  '''
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  '''
  import numpy as np
  import pandas as pd
  model_inputs = np.stack(content_series.map(preprocess))
  model_predictions = model.predict(model_inputs)
  # For some layers, output features will be multi-dimensional tensors.
  # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
  model_output = [model_prediction.flatten() for model_prediction in model_predictions]
  return pd.Series(model_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Input features
# MAGIC As it would be terribly inneficient to deserialize our pre-trained network for each and every record, we make use of a `pandas_udf` function to load model only once per executor. Featurizing our input data consists in a single spark SQL operation

# COMMAND ----------

from typing import Iterator
from pyspark.sql.functions import pandas_udf

@pandas_udf('array<float>')
def vectorizer(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    This loads the model once and then re-use it, amortizing the overhead of loading big models
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param binary_series: This argument is an iterator over batches of data, where each batch
                          is a pandas Series of image data.
    '''
    model = load_model()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

# COMMAND ----------

feature_df = (
  input_df
    .withColumn('features', vectorizer('content'))
    .select('name', 'content', 'label', 'features')
)

display(feature_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model building
# MAGIC With our binary data properly converted as input vectors, we can get to the actual modelling exercise. We will be splitting our feature set in training and testing sets and train multiple `XGBoost` models in parallel, all captured through MLFlow. The goal is to delegate hyperparameter tuning to `Hyperopt` library to get our best performing model programmatically.

# COMMAND ----------

df = feature_df.toPandas()
X = df['features'].tolist()
y = df['label'].tolist()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter tuning
# MAGIC Since all experiments will be tried in parallel, we make our training and testing set accessible from each of our executors. This prevents our driver to send same information prior to any model experiment.

# COMMAND ----------

X_train_B = sc.broadcast(X_train)
y_train_B = sc.broadcast(y_train)

X_test_B = sc.broadcast(X_test)
y_test_B = sc.broadcast(y_test)

# COMMAND ----------

import numpy as np

def to_binary(xs):
  return np.rint(xs).astype(int)

# COMMAND ----------

from hyperopt import hp, tpe, fmin, STATUS_OK, SparkTrials
from xgboost import XGBRegressor
from sklearn import metrics

def train_model(params):
  
  # set up our XGBoost model based on input parameters
  model = XGBRegressor(
    learning_rate = params['learning_rate'],
    max_depth = int(params['max_depth']),
    min_child_weight = int(params['min_child_weight']),
    colsample_bytree = params['colsample_bytree'],
    subsample = params['subsample'],
    n_estimators = int(params['n_estimators'])
  )
  
  # access our training / testing set
  X_local = X_train_B.value
  y_local = y_train_B.value
  X_local_test = X_test_B.value
  y_local_test = y_test_B.value
  
  # train model
  model.fit(X_local, y_local)
  
  # minimizing loss at each step
  pred = model.predict(X_local_test)
  loss = np.sqrt(metrics.mean_squared_error(y_local_test, pred))
  accuracy = metrics.accuracy_score(to_binary(y_local_test), to_binary(pred))
  wsse = metrics.mean_squared_error(y_test_B.value, pred)
  
  # log metrics
  mlflow.log_metric('accuracy', accuracy)
  mlflow.log_metric('wsse', wsse)
  
  return {'status': STATUS_OK, 'loss': loss}

# COMMAND ----------

# define our hyperopts parameters
search_space = {
    'learning_rate':      hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':          hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight':   hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree':   hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':          hp.uniform('subsample', 0.8, 1),
    'n_estimators':       100
}

# execute hyper parameter tuning in parallel
spark_trials = SparkTrials(parallelism=config['ml']['executors'])

# COMMAND ----------

import mlflow

with mlflow.start_run(run_name='classification') as run: 
  run_id = run.info.run_id
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=config['ml']['evals'], 
    trials=spark_trials, 
    rstate=np.random.default_rng(123)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC As represented below, we can access and compare each experiment from the MLFlow console

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://raw.githubusercontent.com/databricks-industry-solutions/car-classification/main/images/car_vision_mlflow.png>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model candidate
# MAGIC We ran multiple experiments for different input parameters. However, we did not log our actual model binary. We can easily train a new model with our best hyper parameters and log content back to MLFlow under a dedicated run.

# COMMAND ----------

best_params

# COMMAND ----------

with mlflow.start_run(run_name='classification-best') as run: 
  
  run_id = run.info.run_id
  
  # Train model with best parameters
  model = XGBRegressor(
    learning_rate = np.arange(0.05, 0.31, 0.05)[best_params['learning_rate']],
    max_depth = np.arange(5, 16, 1, dtype=int)[best_params['max_depth']],
    min_child_weight = np.arange(1, 8, 1, dtype=int)[best_params['min_child_weight']],
    colsample_bytree = np.arange(0.3, 0.8, 0.1)[best_params['colsample_bytree']],
    subsample = best_params['subsample'],
    n_estimators = 100
  )
  
  # Log model
  model.fit(X_train_B.value, y_train_B.value)
  mlflow.xgboost.log_model(model, 'model')
  
  # Compute metrics
  y_pred = model.predict(X_test_B.value)  
  accuracy = metrics.accuracy_score(to_binary(y_test), to_binary(y_pred))
  wsse = metrics.mean_squared_error(y_test_B.value, y_pred)

  # Log metrics
  mlflow.log_metric('accuracy', accuracy)
  mlflow.log_metric('wsse', wsse)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model validation
# MAGIC We can validate our model using simple visualizations that we store on MLFlow against our actual model binary, input parameters and metrics. This context will be useful when reviewing / auditing our approach.

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# retrieve predicted vs. actual value
y_pred = model.predict(list(X_test))
pred_df = pd.DataFrame(zip(y_test, to_binary(y_pred), y_pred), columns=['label', 'prediction', 'probability'])

# create confusion matrix
confusion_matrix = pd.crosstab(pred_df['label'], pred_df['prediction'], rownames=['label'], colnames=['prediction'])

# plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt='d')
plt.savefig("/tmp/confusion_matrix.png")
plt.show()

# COMMAND ----------

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

plt.figure(figsize = (8,5))

# plot model roc_curve
fpr, tpr, _ = roc_curve(pred_df.label, pred_df.probability)
plt.plot(fpr, tpr)

# create our baseline model
ns_probs = [0 for _ in range(pred_df.shape[0])]
ns_fpr, ns_tpr, thresholds = roc_curve(pred_df.label, ns_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--')

plt.title('Receiver operating characteristic curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.savefig("/tmp/roc_curve.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Actionable models
# MAGIC We demonstrated that we could successfully build a regression model that could detect the severity of a given car accident. In this section, we would like to make use of that model in a real life scenario, hence moving from a ML model to some self contained business logic. Using Pyfunc, we will wrap all the necessary context required to classify input images through simple API. 

# COMMAND ----------

class XGBWrapper(mlflow.pyfunc.PythonModel):

  def __init__(self, xgboost):
    self.xgboost = xgboost

  def load_context(self, context): 
    from tensorflow.keras.applications.resnet50 import ResNet50
    resnet = ResNet50(include_top = False)
    self.resnet = resnet

  def __preprocess_image(self, content):
    from PIL import Image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    import io
    img = Image.open(io.BytesIO(content))
    image_size = img.size
    width = image_size[0]
    height = image_size[1]
    if(width != height):
      bigside = width if width > height else height
      background = Image.new('RGB', (bigside, bigside), 'black')
      offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))
      background.paste(img, offset)
      img = background.resize([224, 224])
    else:
      img = img.resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

  def __featurize_series(self, content_series):
    import numpy as np
    import pandas as pd
    model_inputs = np.stack([self.__preprocess_image(x) for x in content_series])
    model_predictions = self.resnet.predict(model_inputs)
    model_output = [model_prediction.flatten() for model_prediction in model_predictions]
    return model_output

  def predict(self, context, content_series):
    input_processed = self.__featurize_series(content_series)
    return self.xgboost.predict(input_processed)

# COMMAND ----------

import xgboost as xgb
import sklearn
import tensorflow as tf
import PIL

with mlflow.start_run(run_name='classification-wrapper'):

  conda_env = mlflow.pyfunc.get_default_conda_env()
  conda_env['dependencies'][2]['pip'] += ['xgboost=={}'.format(xgb.__version__)]
  conda_env['dependencies'][2]['pip'] += ['sklearn=={}'.format(sklearn.__version__)]
  conda_env['dependencies'][2]['pip'] += ['tensorflow=={}'.format(tf.__version__)]
  conda_env['dependencies'][2]['pip'] += ['pillow=={}'.format(PIL.__version__)]
  conda_env['dependencies'][2]['pip'] += ['pandas=={}'.format(pd.__version__)]
  conda_env['dependencies'][2]['pip'] += ['numpy=={}'.format(np.__version__)]
  
  mlflow.pyfunc.log_model(
    'pipeline', 
    python_model=XGBWrapper(model), 
    conda_env=conda_env
  )
  
  api_run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# we do not wish to re-open run_id and therefore alter start / end time. 
# instead, we log artifact to existing run
client = mlflow.tracking.MlflowClient()

# we demonstrate that our submitted model was the best fit
client.log_artifact(api_run_id, "/tmp/roc_curve.png")
client.log_artifact(api_run_id, "/tmp/confusion_matrix.png")

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_name = config['ml']['name']
model_uri = 'runs:/{}/pipeline'.format(api_run_id)
result = mlflow.register_model(model_uri, model_name)
version = result.version

# COMMAND ----------

# MAGIC %md
# MAGIC We can also promote our model to different stages programmatically. Although our models would need to be reviewed in real life scenario, we make it available as a production artifact for our next notebook and programmatically transition previous runs back to Archive.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
for model in client.search_model_versions("name='{}'".format(model_name)):
  if model.current_stage == 'Production':
    print("Archiving model version {}".format(model.version))
    client.transition_model_version_stage(
      name=model_name,
      version=int(model.version),
      stage="Archived"
    )

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model inference
# MAGIC We now have a pipeline that can be executed to infer severity of a car accident given a collection of pictures. Although this model may be surfaced via MLFlow serving API, we demonstrate usage from a pandas dataframe for the purpose of this solution.

# COMMAND ----------

input_df = (
  spark
    .read
    .format('binaryFile')
    .load('{}/accidents'.format(config['input_dir']))
    .toPandas()
)

# COMMAND ----------

wrapper = mlflow.pyfunc.load_model("runs:/{}/pipeline".format(api_run_id))
input_df['severity'] = wrapper.predict(input_df['content'])

# COMMAND ----------

# MAGIC %md
# MAGIC We can extract pictures at given thresholds of severity to visualize our model output. Although our model seemed to work extremely well, we can observe some obvious misclassification in our approach. However, we believe that this is due to the low quality of input data at our disposal for this example and expect this model to work much better against real life claim pictures previously labelled and review. 

# COMMAND ----------

bins = [0, 0.4, 0.6, 1.5]
input_df['bins'] = pd.cut(x=input_df['severity'], bins=bins, labels=['minor', 'medium', 'major'])

# COMMAND ----------

from PIL import Image
import io

def plot_accidents(severity):
  df = input_df[input_df['bins'] == severity].sort_values(by='severity')
  plt.figure(figsize=(20,5))
  i = 1
  for _, f in list(df.iterrows())[:3]:
    img = Image.open(io.BytesIO(f.content))
    s = plt.subplot(1, 3, i)
    s.set_xlabel(f.path.split('/')[-1])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(img)
    i = i + 1

# COMMAND ----------

plot_accidents('minor')

# COMMAND ----------

# MAGIC %md
# MAGIC In this solution, we did not demonstrate how to train the best model for computer vision in the context of claim automation but rather to lay down a framework that can be used to accelerate this complex journey. We recommend organizations to spend enough time upfront collecting a good enough set of input images and properly classify the severity of each accident to yield better predictions. 
