# Titanic Passenger Survival Prediction
SKLearn KneighborsClassifier model that predicts a Titanic passenger's survival

# Data
My class provided a dataset with information about the titanic passengers and their info.
Variable	Definition	
survival	Survival	
pclass	    Ticket class	
sex	        Sex	
Age	        Age in years	
sibsp	    # of siblings / spouses aboard the Titanic	
parch	    # of parents / children aboard the Titanic	
ticket	    Ticket number	
fare	    Passenger fare	
cabin	    Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

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



