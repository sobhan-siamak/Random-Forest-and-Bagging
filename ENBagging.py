


##############@copy by sobhan siamak ##########



import pandas as  pd
import  numpy as np
import  matplotlib.pyplot as plt
from  sklearn.ensemble import BaggingClassifier
from  sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


#Slicing Example in Python
# a=np.array([[1,2,3,4,5,6,7,8],[4,5,6,5,4,3,2,1]])
# print(a[:,:7])
# print(a[:,7])



# importing the Datasets
#Heart= pd.read_csv('Heart.txt')
Heart=np.loadtxt('Heart.txt')#have 14 columns
XHeart= Heart[:,:13]
YHeart= Heart[:,13]
#Glass= pd.read_csv('Glass.txt')
Glass= np.loadtxt('Glass.txt')#have 10 columns
XGlass= Glass[:,:9]
YGlass= Glass[:,9]
#Sonar= pd.read_csv('Sonar.txt')
Sonar= np.loadtxt('Sonar.txt')#have 61 columns
XSonar= Sonar[:,:60]
YSonar= Sonar[:,60]
#Ionosphere= pd.read_csv('Ionosphere.txt')
Ionosphere= np.loadtxt('Ionosphere.txt')#have 35 column
XIonosphere= Ionosphere[:,:34]
YIonosphere= Ionosphere[:,34]
#ColonTumor= pd.read_csv('Colon Tumor.txt')
ColonTumor= np.loadtxt('Colon Tumor.txt')#have 2001 columns
XColonTumor= ColonTumor[:,:2000]
YColonTumor= ColonTumor[:,2000]
#ConcentericRectangles= pd.read_csv('Concentric_rectangles.txt')
ConcentericRectangles= np.loadtxt('Concentric_rectangles.txt')#have 3 columns
XConcentericRectangles= ConcentericRectangles[:,:2]
YConcentericRectangles= ConcentericRectangles[:,2]


# Splitting the datasets into the Training sets and Test sets



# #1-Heart Dataset
# X_Train, X_Test, Y_Train, Y_Test= train_test_split(XHeart, YHeart,test_size=0.30, random_state=0)
# #2-Glass Dataset
# #X_Train, X_Test, Y_Train, Y_Test= train_test_split(XGlass, YGlass,test_size=0.30, random_state=0)
# #3-Sonar Dataset
# #X_Train, X_Test, Y_Train, Y_Test= train_test_split(XSonar, YSonar,test_size=0.30, random_state=0)
# #4-Ionosphere Dataset
# #X_Train, X_Test, Y_Train, Y_Test= train_test_split(XIonosphere, YIonosphere,test_size=0.30, random_state=0)
# #5-ColonTomor Dataset
# #X_Train, X_Test, Y_Train, Y_Test= train_test_split(XColonTumor, YColonTumor,test_size=0.30, random_state=0)
# #6-Concentric Dataset
# #X_Train, X_Test, Y_Train, Y_Test= train_test_split(XConcentericRectangles, YConcentericRectangles,test_size=0.30, random_state=0)
#
#
#
# BGclassifier= BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.1, n_estimators=20)
# BGclassifier.fit(X_Train,Y_Train)
# ACC=BGclassifier.score(X_Test,Y_Test)
# print(ACC)
#
#


ACC=0
a=np.zeros(10)
for i in range(10):
    # 1-Heart Dataset
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(XHeart, YHeart, test_size=0.30, random_state=0)
    # 2-Glass Dataset
    # X_Train, X_Test, Y_Train, Y_Test= train_test_split(XGlass, YGlass,test_size=0.30, random_state=0)
    # 3-Sonar Dataset
    # X_Train, X_Test, Y_Train, Y_Test= train_test_split(XSonar, YSonar,test_size=0.30, random_state=0)
    # 4-Ionosphere Dataset
    # X_Train, X_Test, Y_Train, Y_Test= train_test_split(XIonosphere, YIonosphere,test_size=0.30, random_state=0)
    # 5-ColonTomor Dataset
    # X_Train, X_Test, Y_Train, Y_Test= train_test_split(XColonTumor, YColonTumor,test_size=0.30, random_state=0)
    # 6-Concentric Dataset
    # X_Train, X_Test, Y_Train, Y_Test= train_test_split(XConcentericRectangles, YConcentericRectangles,test_size=0.30, random_state=0)

    BGclassifier = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5, n_estimators=20)
    BGclassifier.fit(X_Train, Y_Train)
    accur = BGclassifier.score(X_Test, Y_Test)
    ACC=ACC+accur
    a[i]=ACC
    #if i==10:



print("The Average of 10 Run Accuracy is: ")
print(ACC/10)
print(np.std(a))
print(a)









