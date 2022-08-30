<img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fs-lakehouse-logo.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-10.4ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/10.4ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
[![POC](https://img.shields.io/badge/POC-5_days-green?style=for-the-badge)](https://databricks.com/try-databricks)


*With over 15 thousands car accidents in the US every day (10 accidents every minute), automotive insurers recognize the need to improve operational efficiency through the use of AI. Specifically, computer vision has been widely discussed as a way to optimize the claims process from filing to settlement. The reality is that computer vision is complex and typically requires large investments in term of both resources and infrastructure. By applying transfer learning on pre-trained neural networks, we demonstrate how Databricks helps insurance companies kickstart their AI/Computer Vision journey towards claim assessment and damage estimation.*

___
<antoine.amend@databricks.com>

___

<img src=https://raw.githubusercontent.com/databricks-industry-solutions/car-classification/main/images/reference_architecture.png alt="logical_flow" width="800">

___ 

Using computer vision, automotive insurers could estimate the costs of damage to a car given a simple picture uploaded by their policy holders. 
Whilst the task at hand would be complex on its own (yet achievable), it is impossible to fulfill without first having access to a database of historical claims data, their estimated costs and uploaded pictures. As we do not have access to such a database, we started to look for alternative datasets available on the internet. Our first approach was to look at Stanford car [dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) but quickly realised we would need its equivalent for damaged vehicles together with damage estimate, so we decided to bring a more creative and pragmatic approach to problem solving.

Using pictures of cars accidents taken from the internet (manually scraped from Google image), we believe that we could train a binary classifier that learns how to differentiate damaged (labelled 1) from non damaged cars (labelled 0). Using an existing deep learning model that convert raw images into input feature vectors, our model would then "learn" the shapes that no longer "look like" a car, returning a probability distribution instead of a simple binary output. We assume that a simple dent on a bumper or a few paint scratches would be scored low compare to a more critical accident with half of the engine gone, hence quantifying the severity of an accident as a proxy for damaged value and leaving our readers with an open ended question: **how expensive is a 20% damaged car to fix?**. This would help insurers triage critical from minor events or automatically dispatch replacement vehicles.

<br>

<img src=https://raw.githubusercontent.com/databricks-industry-solutions/car-classification/main/images/car_vision_classification.png width="600">

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                               | description             | license    | source                                              |
|-------------------------------------------------------|-------------------------|------------|-----------------------------------------------------|
| tensorflow                                            | Tensorflow              | Apache2    | https://www.tensorflow.org/                         |
| Keras                                                 | Tensorflow abstraction  | MIT        | https://github.com/keras-team/keras                 |
| Pillow                                                | Image processing        | HPND       | https://python-pillow.org/                          |
| PyYAML                                                | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |

