import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd
import seaborn as sns

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("titanic_model/titanic.csv")

#print(titanic_df["Sex"].value_counts(dropna=False))


# Fill in missing data in Age feature
mask = (titanic_df['Pclass'] == 1) & (titanic_df['Sex'] == 'male')
avg_filler = titanic_df.loc[mask, 'Age'].mean()
titanic_df.loc[titanic_df['Age'].isnull() & mask, 'Age'] = avg_filler

mask = (titanic_df['Pclass'] == 2) & (titanic_df['Sex'] == 'male')
avg_filler = titanic_df.loc[mask, 'Age'].mean()
titanic_df.loc[titanic_df['Age'].isnull() & mask, 'Age'] = avg_filler

mask = (titanic_df['Pclass'] == 3) & (titanic_df['Sex'] == 'male')
avg_filler = titanic_df.loc[mask, 'Age'].mean()
titanic_df.loc[titanic_df['Age'].isnull() & mask, 'Age'] = avg_filler

mask = (titanic_df['Pclass'] == 1) & (titanic_df['Sex'] == 'female')
avg_filler = titanic_df.loc[mask, 'Age'].mean()
titanic_df.loc[titanic_df['Age'].isnull() & mask, 'Age'] = avg_filler

mask = (titanic_df['Pclass'] == 2) & (titanic_df['Sex'] == 'female')
avg_filler = titanic_df.loc[mask, 'Age'].mean()
titanic_df.loc[titanic_df['Age'].isnull() & mask, 'Age'] = avg_filler

mask = (titanic_df['Pclass'] == 3) & (titanic_df['Sex'] == 'female')
avg_filler = titanic_df.loc[mask, 'Age'].mean()
titanic_df.loc[titanic_df['Age'].isnull() & mask, 'Age'] = avg_filler
#print(titanic_df.isnull().sum())

titanic_df.drop("Cabin", inplace=True, axis=1)
titanic_df["Embarked"].fillna(value="U", inplace=True)

# Categorical data converted to numerical
titanic_df["Embarked"].replace(["U","S","C","Q"],[0,1,2,3], inplace=True)
titanic_df["Sex"].replace(["male","female"],[1,2], inplace=True)

titanic_df["Ticket"] = titanic_df["Ticket"].str.extract(r'(\d+)', expand=False)
titanic_df["Ticket"] = pd.to_numeric(titanic_df['Ticket'])


#print(titanic_df["Ticket"])


titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["PassengerId"],
    description="Titanic dataset")
titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

#expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="iris_dimensions")    
#value_between(expectation_suite, "sepal_length", 4.5, 8.0)
#value_between(expectation_suite, "sepal_width", 2.1, 4.5)
#value_between(expectation_suite, "petal_length", 1.2, 7)
#value_between(expectation_suite, "petal_width", 0.2, 2.5)
#iris_fg.save_expectation_suite(expectation_suite=expectation_suite, validation_ingestion_policy="STRICT")    
    

