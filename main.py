import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")




# feature selection
features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])
y = train_data['Survived']

# split data to train set and validation set
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Random forest prediction
forest_model = RandomForestClassifier(n_estimators=100)
forest_model.fit(train_X, train_y)
forest_val_predictions = forest_model.predict(val_X)
print('MAE of Random Forests: %f' % (mean_absolute_error(val_y, forest_val_predictions)))



# Decision Tree prediction
tree_model = DecisionTreeRegressor(random_state=1)
tree_model.fit(train_X, train_y)
tree_val_predictions = tree_model.predict(val_X)
print('MAE of Decision Tree: %f' % (mean_absolute_error(val_y, tree_val_predictions)))


# testing
X_test = pd.get_dummies(test_data[features])
predictions = forest_model.predict(X_test)

# generate CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print('csv generated.')


