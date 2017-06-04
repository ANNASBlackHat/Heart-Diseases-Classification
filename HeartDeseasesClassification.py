from sklearn import tree

clf = tree.DecissionTreeClassifier()

#[age, gender, chect pain, blood plessure]
# Change dataset to number
# Gender : 1 (male), 0 (female)
# Chest Pain : 1 (typical type), 2 (typical type angina), 3 (non-angina pain), 4 (asymptomatic)
# Blood Pleassure : High (<=120 mg/dl), Very High(>=120 mg/dl)
X = [[63,1,2,115],[67,1,4,125],[67,1,4,116],[37,1,3,117],[41,0,1,118],[56,1,1,119],
    [62,0,4,119],[57,0,4,115],[63,1,4,114],[53,1,4,116],[57,1,4,114],[56,0,1,117],
    [56,1,3,115],[44,1,1,117]]

Y = ['No','Yes','Yes','No','No','No','Yes','No','Yes','Yes','No','No','Yes','No']

clf = clf.fit(X, Y)

prediction = clf.predict([[65,1,4,116]])

print(prediction)
