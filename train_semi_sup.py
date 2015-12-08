# import
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn.ensemble as se
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


# function to write submission output file
def outputSubmission(model,name):
    print "making predictions ......"
    predicted = model.predict(test)
    # write
    print "writing submission file ......"
    w = open("submission/"+name+".csv","w")
    w.write("Id,Cover_Type\n")
    test_ids = test_id
    for i in range(len(predicted)):
        w.write(str(test_ids[i])+","+str(predicted[i])+"\n")
    w.close()
    
    
# function to calculate fitting and prediction error
def errFn(y,yhat):
    diff = np.array(y)-np.array(yhat)
    binary = diff != 0
    return float(sum(binary)) / len(y)


# data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train_label = train["Cover_Type"]
train_id = train["Id"]
train.__delitem__("Cover_Type")
train.__delitem__("Id")
test_id = test["Id"]
test.__delitem__("Id")

# transform 2-way interaction
cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',
       'Hillshade_9am','Hillshade_Noon','Hillshade_3pm']

for c1 in cols:
    for c2 in cols:
        if (c1+"."+c2 not in train.columns) and (c2+"."+c1 not in train.columns):
            train[c1+"."+c2] = train[c1]*train[c2]
            test[c1+"."+c2] = test[c1]*test[c2]

print train.shape
print test.shape


rate = 1

train_comb = train
# select 10% of test data
selected_test = test.sample(test.shape[0]/rate,replace=False,random_state=20422438)
train_comb = train_comb.append(selected_test)
train_comb_label = train_label
train_comb_unlabeled = pd.DataFrame(np.array([-1]*(test.shape[0]/rate)))
train_comb_label = np.array(train_comb_label.append(train_comb_unlabeled))
train_comb_label = train_comb_label.reshape(len(train_comb_label))


a_level = 1
label_prop_model = LabelSpreading(kernel="knn",alpha=a_level)
label_prop_model.fit(train_comb, train_comb_label)
pred_y = label_prop_model.transduction_
pred_y[:train.shape[0]] = train_label



X_train, X_test, y_train, y_test = train_test_split(label_prop_model.X_, pred_y, test_size=0.10, random_state=20422438)

model_erf = se.ExtraTreesClassifier(random_state=20422438,n_jobs=-1,n_estimators=1000)
model_erf.fit(X_train,y_train)

model_erf_pred = model_erf.predict(X_test)
model_erf_error = errFn(model_erf_pred,y_test)

print a_level
print model_erf_error

outputSubmission(model_erf,"semi_sup_model_int2_prop_erf_100perc_alpha1")

