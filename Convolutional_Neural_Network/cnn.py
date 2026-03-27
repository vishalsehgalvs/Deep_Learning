import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron  # Used for simple linear classification tasks.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential  # Sequential lets you build a neural network layer-by-layer in Keras.

from tensorflow.keras.layers import Dense  # Dense makes the final predictions
from tensorflow.keras.layers import Conv2D  # Conv2D extracts features
from tensorflow.keras.layers import Flatten  # Flatten reshapes them

from tensorflow.keras.layers import MaxPooling2D  # MaxPooling2D reduces size
from tensorflow.keras.layers import Dropout  # Dropout prevents overfitting

from tensorflow.keras.utils import \
    to_categorical  # converts numeric class labels into one-hot encoded format for training classification models
from keras.datasets import mnist  # big data set

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv("test.csv")
# print(df_train.head())
#    label  pixel0  pixel1  pixel2  ...  pixel780  pixel781  pixel782  pixel783
# 0      1       0       0       0  ...         0         0         0         0
# 1      0       0       0       0  ...         0         0         0         0
# 2      1       0       0       0  ...         0         0         0         0
# 3      4       0       0       0  ...         0         0         0         0
# 4      0       0       0       0  ...         0         0         0         0
#
# [5 rows x 785 columns]

# print(df_train.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 42000 entries, 0 to 41999
# Columns: 785 entries, label to pixel783
# dtypes: int64(785)
# memory usage: 251.5 MB

# print(df_test.isnull())
#        pixel0  pixel1  pixel2  pixel3  ...  pixel780  pixel781  pixel782  pixel783
# 0       False   False   False   False  ...     False     False     False     False
# 1       False   False   False   False  ...     False     False     False     False
# 2       False   False   False   False  ...     False     False     False     False
# 3       False   False   False   False  ...     False     False     False     False
# 4       False   False   False   False  ...     False     False     False     False
# ...       ...     ...     ...     ...  ...       ...       ...       ...       ...
# 27995   False   False   False   False  ...     False     False     False     False
# 27996   False   False   False   False  ...     False     False     False     False
# 27997   False   False   False   False  ...     False     False     False     False
# 27998   False   False   False   False  ...     False     False     False     False
# 27999   False   False   False   False  ...     False     False     False     False
#
# [28000 rows x 784 columns]

# print(df_test.isnull().sum())
# pixel0      0
# pixel1      0
# pixel2      0
# pixel3      0
# pixel4      0
#            ..
# pixel779    0
# pixel780    0
# pixel781    0
# pixel782    0
# pixel783    0
# Length: 784, dtype: int64

#  preprocessing the data

X_train = df_train.drop("label", axis=1).value
y_train = df_train['label'].value
X_test = df_test.drop('label",axis =1').value
y_test = df_test['label'].value

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
