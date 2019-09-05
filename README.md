# ML Project 2 Recommender System

## Team Members

* Asli Yorusun : asli.yorusun@epfl.ch
* Erdem Bocugoz : erdem.bocugoz@epfl.ch
* Serif Soner Serbest : serif.serbest@epfl.ch

## CrowdAI Best Score

Name : Soner
Submission ID : 23831

We managed to be in top 20 by scoring 1.018 RMSE where the best score of the competition is 1.016 RMSE.
You can find the competition link [here](https://www.crowdai.org/challenges/epfl-ml-recommender-system/leaderboards).

## Project Report
You can find the detailed project report [here](https://github.com/serifsonerserbest/Recommender-System/blob/master/Recommender_System_Project_Report.pdf).

## Required Libraries and Setting up the Envioronment 

* We used python evironment for our project (anaconda).

pip install libraries
* scikit-learn
* Pandas
* NumPy
* Pickle
* scipy
* os

Install custom libraries
* Surprise
  * You can find detailed installation guide and how to use the library here: http://surpriselib.com
  * Or you can directly install it from this [this GitHub repo](https://github.com/NicolasHug/Surprise) by:
```
 git clone https://github.com/NicolasHug/surprise.git
 python setup.py install
```

* PyFM
  * You can directly install it from this [this GitHub repo](https://github.com/coreylynch/pyFM) by:
```
 pip install git+https://github.com/coreylynch/pyFM
```

* PySpark
  * You can find a detailed installation guide for pySpark [here](https://medium.com/tinghaochen/how-to-install-pyspark-locally-94501eefe421).

* Data Sets:
  * Data sets are taken from [here](https://www.crowdai.org/challenges/epfl-ml-recommender-system/dataset_files).
  note that you need a epfl e-mail address to reach web site.
  
## Files

* Data Files : 
  * data_train.csv : train set
  * data_test.csv : test set provided (originally sampleSubmissionn.csv)
  * tmp_train.csv : train file obtained from train-test split of train set
  * tmp_test.csv : test file obtained from train-test split of train set
* Python files :
  * model_surprise.py : Contains the models from Surprise library : BaseLineOnly, SlopeOne, KNN, SVD, SVD++
  * model_pyfm.py : Contains the model from PyFM library (FM refers to Factorization Machine)
  * model_pyspark.py Contains the model from PySpark library : ALS
  * model_matrixfactorization.py : Contains the models we implemented by ourselves using Exercise 10 template: SGD, ALS
  * model_means.py : Contains the models we implemented by ourselves  Global Mean, User Mean, Item Mean
  * matrix_fact_helpers : Helper functions for the models we implemented from Exerice 10 
  * hyperparameter_tuning.py : Contains functions for hyper parameter tuning for most of the models.
  * blend.py : Contains blending(voting) function to ensemble different models.
  * implementations.py : Contains helper functions, i.e. reading csv files, transforming data frames etc.
* Pickle Object:
* linreg.pkl : linear regression model for blending
