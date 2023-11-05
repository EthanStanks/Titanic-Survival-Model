# Titanic Passenger Survival Prediction
SKLearn KneighborsClassifier model that predicts a Titanic passenger's survival

# Data
My class provided a dataset with information about the titanic passengers and their info.
Features in the dataset: Survival, Ticket class, Sex, Age in years, # of family members, Ticket number, Passenger fare, Cabin number, and Port of Embarkation.

# Data Processing
I dropped all n/a values as well as multiple features that didn't correlate to a passenger's survival.
As well as some data mapping and rounding floats to ints.

# EDA
EDA was performed by creating graphs with matplotlib and seaborn.
Graphs include visualing the correlation of data and class balances.

# Model
Train and test data was created using SKLearn's train_test_split()
Model is trained and fitted using SKLearn's KneighborsClassifier.

# Validation
Model is scored on test and train using SKLearn's score()
A cross validation score is obtained using SKLearn's cross_val_score()
A classification report is obtained using SKLearn's classification_report()
A heatmap of the confusion matrix from the model's prediction is created using seaborn.



