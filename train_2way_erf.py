# import
import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn.ensemble as se

# train data
train = pd.read_csv("data/train.csv")
y = train["Cover_Type"]
train.__delitem__("Id")
train.__delitem__("Cover_Type")

# test data
test = pd.read_csv("data/test.csv")
test_ids = test["Id"]
test.__delitem__("Id")

cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',
       'Hillshade_9am','Hillshade_Noon','Hillshade_3pm']

# interaction
train_interaction = train
test_interaction = test
for c1 in cols:
    for c2 in cols:
        if (c1+"."+c2 not in train_interaction.columns) and (c2+"."+c1 not in train_interaction.columns):
            train_interaction[c1+"."+c2] = train[c1]*train[c2]
            test_interaction[c1+"."+c2] = test[c1]*test[c2]

print train_interaction.shape
print test_interaction.shape

# model
model_best_erf = se.ExtraTreesClassifier(random_state=20422438,n_jobs=-1,n_estimators=1500,min_samples_split=2,
                                         criterion="gini")
model_best_erf.fit(train_interaction,y)

# predict
model_best_erf_pred = model_best_erf.predict(test_interaction)

# submission
w = open("submission/"+"2_interaction_erf_best"+".csv","w")
w.write("Id,Cover_Type\n")
for i in range(len(model_best_erf_pred)):
    w.write(str(test_ids[i])+","+str(model_best_erf_pred[i])+"\n")
w.close()

