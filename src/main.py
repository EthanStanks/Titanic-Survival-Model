import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    data_folder = 'data/'
    data_file = 'titanic_passengers.csv'
    data_path = os.path.join(data_folder, data_file)
    output_folder = 'output/'
    df = pd.read_csv(data_path)
    seed = 23441
    perform_eda = True
    perform_ml = True

    # Data Processing
    df.info()
    df = df.drop('Name', axis=1) #useless info
    df = df.drop('Ticket', axis=1) #useless info
    df = df.drop('Cabin', axis=1) #data is awful
    df = df.drop('PassengerId', axis=1) #no correlation to anything
    df = df.dropna(axis=0)
    sex_mapping = {'male': 0,'female': 1}
    embark_mapping = {'Q': 0, 'C': 1, 'S': 2}
    df['D_Sex'] = df['Sex'].map(sex_mapping)
    df['D_Embarked'] = df['Embarked'].map(embark_mapping).astype('int64')
    df['Family'] = df['SibSp'] + df['Parch']
    df = df.drop(['SibSp', 'Parch'], axis=1)
    df['Age'] = df['Age'].round(0).astype('int64')
    df = df[df['Age'] >= 1]
    df['Fare'] = df['Fare'].round(0).astype('int64')
    df.info()

    # EDA
    if(perform_eda):
        # correlation of the data
        ax = sns.heatmap(df.corr(numeric_only=True), annot=True)
        corr_clean = os.path.join(output_folder, 'corr_clean_plot.png')
        plt.savefig(corr_clean)
        plt.close()

        # draw bar graph for ticket class balance
        type_counts = df['Pclass'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.bar(type_counts.index, type_counts.values)
        plt.xlabel('Ticket Class')
        plt.ylabel('Number of Samples')
        plt.title('Balance of Ticket Class')
        plt.xticks(type_counts.index, ['1', '2', '3'])
        ticket_class_balance_bar_path = os.path.join(output_folder, 'bar_ticket_class_balance.png')
        plt.savefig(ticket_class_balance_bar_path)
        plt.close()

        # draw bar graph for sex balance
        type_counts = df['Sex'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.bar(type_counts.index, type_counts.values)
        plt.xlabel('Sex')
        plt.ylabel('Number of Samples')
        plt.title('Balance of Sex')
        plt.xticks(type_counts.index, ['male', 'female'])
        sex_balance_bar_path = os.path.join(output_folder, 'bar_sex_balance.png')
        plt.savefig(sex_balance_bar_path)
        plt.close()

        # draw bar graph for survival balance
        type_counts = df['Survived'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.bar(type_counts.index, type_counts.values)
        plt.xlabel('Survived')
        plt.ylabel('Number of Samples')
        plt.title('Balance of Survival Rate')
        plt.xticks(type_counts.index, ['No', 'Yes'])
        survived_balance_bar_path = os.path.join(output_folder, 'bar_survived_balance.png')
        plt.savefig(survived_balance_bar_path)
        plt.close()

        # draw bar graph for age balance
        type_counts = df['Age'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.bar(type_counts.index, type_counts.values)
        plt.xlabel('Age')
        plt.ylabel('Number of Samples')
        plt.title('Balance of Ages')
        age_balance_bar_path = os.path.join(output_folder, 'bar_age_balance.png')
        plt.savefig(age_balance_bar_path)
        plt.close()

    # guess survival based off Pclass, Sex, and Age
    if(perform_ml):

        # Train Model 
        X = df.loc[:, ['D_Sex', 'Pclass']]
        y = df.loc[:, ['Survived']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)

        # Model Validation 
        male = 0
        female = 1
        new_data = [[male, 2]]
        prediction = model.predict(new_data)
        if (prediction == 1): print("Prediction: Survived")
        else: print("Prediction: Died")

        score = model.score(X_test, y_test)
        print("Test Score",score)

        score = model.score(X_train, y_train)
        print("Train Score",score)

        cross_val_scores = cross_val_score(model, X, y, cv=10)
        print("Cross-Validated Scores:", cross_val_scores)

        print(classification_report(y_test, model.predict(X_test)))

        cmat = confusion_matrix(y, model.predict(X))
        sns.heatmap(cmat, annot=True, fmt='d')
        confusion_matrix_path = os.path.join(output_folder, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
    
    

    

