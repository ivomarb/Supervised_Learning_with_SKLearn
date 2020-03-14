## The task
Develop a classifier for the following public data set. [Sample data (1K)](https://drive.google.com/file/d/10CNTepw2ZhuBfVD8mStm89Imo4xJcWd0/view?usp=sharing), [larger sample data (1M)](https://drive.google.com/file/d/1t_x0kzRufbaqyJ3dGTQK-QJO9TFfOu2g/view?usp=sharing).

## The dataset
The dataset is part of this public dataset[[1]](#1).

Each record corresponds to an ad impression served by Adform A/S, and consists of a single binary label (clicked/not-clicked) and a selected subset of features (c0-c9). The positive examples and negative examples are downsampled at different rates. The data is chronologically ordered (top of file = oldest, bottom of file = youngest).

The file is gzipped and each line corresponds to a single record, serialized as JSON. The JSON has the following fields:

 - "l": The binary label indicating whether the ad was clicked (1) or not (0).
 - "c0" - "c9": Categorical features which were hashed into a 32-bit integer.
 
The semantics of the features are not disclosed. The values are stored in an array, because some of the features have multiple values per record. When a key is missing, it means the features are missing.

### Notes
*PLEASE NOTE THAT THE FEATURES ARE CATEGORICAL (IDs), AND NOT NUMERICAL OR ORDINAL!* M Imagine it is something like what product the user has clicked on in the past.
The classifier will periodically be trained and updated, and will be used to predict on future data
 - Feature c6 and c9 are very high dimensional, and there are multiple values for each record
 - All other features (c0 - c5, c7, c8) are scalar (there are only 1 value for each record)



