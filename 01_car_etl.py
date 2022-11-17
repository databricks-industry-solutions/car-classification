# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this notebook at https://github.com/databricks-industry-solutions/car-classification. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Accessing pictures
# MAGIC As we do not have access to an insurance database made of historic claims with both damage (in dollar value) and relevant picture, we started to look at existing datasets on the internet. Out first approach was to look at Stanford car [dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) but quickly realised we would need its "equivalent" dataset for damaged cars. We decided to manually scrape both damaged and undamaged cars from google images, 300 images for each class, and available in DBFS. For legal reasons, we are not able to redistribute this dataset as part of this solution accelerator.

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

accident_dir = dbutils.fs.ls('{}/accidents'.format(config['input_dir']))
accident_pic = list(filter(lambda x: x.name.startswith("accident_"), accident_dir))

car_dir = dbutils.fs.ls('{}/cars'.format(config['input_dir']))
car_pic = list(filter(lambda x: x.name.startswith("car_"), car_dir))

display(accident_dir)

# COMMAND ----------

from PIL import Image

def resize(img):
  image_size = img.size
  width = image_size[0]
  height = image_size[1]
  if(width != height):
    bigside = width if width > height else height
    background = Image.new('RGB', (bigside, bigside), 'black')
    offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))
    background.paste(img, offset)
    return background.resize([224, 224])
  else:
    return img.resize([224, 224])

def get_image(img_path):
  img = Image.open(img_path)
  img = resize(img)
  return img

# COMMAND ----------

# MAGIC %md
# MAGIC Using Pillow library, we can easy ready raw images stored on distributed file storage (storage previously mounted as disk) and render the same through matplotlib visualizations. Let's see what our car dataset looks like.

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
for i, f in enumerate(car_pic[:9]):
    img = get_image(f.path.replace('dbfs:/', '/dbfs/'))
    s = plt.subplot(3, 3, i+1)
    s.set_xlabel(f.path.split('/')[-1])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(img)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resizing pictures
# MAGIC As `ResNet-50` neural network (next notebook) was trained on 224x224 images, we will also need to resize each picture, adding black background if required. This process will be handled at scale, through the use of user defined functions on binary data. Although we demonstrated how to read raw pictures from dbfs, it would be terribly inneficient to do so from an image processing perspective. Instead, we can load fines as binary through spark SQL operations and persist binary data back into delta lake.

# COMMAND ----------

df_acc = spark.read.format('binaryFile').load('{}/accidents'.format(config['input_dir']))
df_car = spark.read.format('binaryFile').load('{}/cars'.format(config['input_dir']))
display(df_acc)

# COMMAND ----------

from pyspark.sql import functions as F
import io

@F.udf('string')
def get_name(path):
  return path.split('/')[-1]

@F.udf('binary')
def resize_image(image_data):
  img = Image.open(io.BytesIO(image_data))
  img = resize(img)
  imgByteArr = io.BytesIO()
  img.save(imgByteArr, format='png')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store to Delta Table
# MAGIC As we will be using our data for ML, we may need to consolidate all our records onto one delta table, resizing pictures to a guarantee compatibility with our convolutional neural network (expect pictures of size 224x224)

# COMMAND ----------

_ = (
  df_acc.union(df_car)
    .withColumn('name', get_name('path'))
    .withColumn('content', resize_image('content'))
    .withColumn('label', F.when(F.col('name').startswith('accident'), F.lit(1)).otherwise(F.lit(0)))
    .select('name', 'content', 'label')
    .write
    .format('delta')
    .saveAsTable(config['database']['tables']['images'], mode="append")
)

# COMMAND ----------

# MAGIC %md
# MAGIC As reported below, our images were correctly resized and binary data was preserved through delta write operations. 

# COMMAND ----------

image_data = spark.read.table(config['database']['tables']['images']).limit(1).toPandas().iloc[0]['content']
img = Image.open(io.BytesIO(image_data))
plt.imshow(img)

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we have created a collection of curated images (resized to 224x224) of car accidents that we can use for machine learning purpose. Using delta as backbone to this solution, we ensure strict governance (time travel) and performance of our model / queries.
