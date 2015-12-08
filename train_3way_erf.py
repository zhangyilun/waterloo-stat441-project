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

train_3_way = train_interaction
test_3_way = test_interaction

# add 3 way interactions
for c1 in cols:
    for c2 in cols:
        for c3 in cols:
            if (cols.index(c1) <= cols.index(c2)) and (cols.index(c2) <= cols.index(c3)):
                train_3_way[c1+"."+c2+"."+c3] = train[c1]*train[c2]*train[c3]
                test_3_way[c1+"."+c2+"."+c3] = test[c1]*test[c2]*test[c3]


print train_3_way.shape
print test_3_way.shape

X_train_3w, X_test_3w, y_train, y_test = train_test_split(train_3_way,y,test_size=0.10,random_state=20422438)

# test model
model_erf = se.ExtraTreesClassifier(random_state=20422438,n_jobs=-1,n_estimators=1000)
model_erf.fit(X_train_3w,y_train)
model_erf_pred = model_erf.predict(X_test_3w)

# feature selection
feat_imp = model_erf.feature_importances_
selected_val = feat_imp[feat_imp > 0.0025]
selected_index = [list(feat_imp).index(x) for x in selected_val]
print len(selected_index)

# best 3-way
model_best_erf = se.ExtraTreesClassifier(n_jobs=-1,n_estimators=1000,criterion="entropy")
model_best_erf.fit(train_3_way[selected_index],y)
model_best_erf_pred = model_best_erf.predict(test_3_way[selected_index])

# submission
w = open("submission/"+"3_interaction_feat_imp_erf_best"+".csv","w")
w.write("Id,Cover_Type\n")
for i in range(len(model_best_erf_pred)):
    w.write(str(test_ids[i])+","+str(model_best_erf_pred[i])+"\n")
w.close()

