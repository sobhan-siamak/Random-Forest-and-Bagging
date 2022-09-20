


############## @copy by sobhan siamak ##########

import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from pprint import pprint

from sklearn.model_selection import train_test_split


######Heart Dataset are 2 classes  and max 2 value for features
######GLASS Dataset are 6 classes  and max 4 value for features
######Sonar Dataset are 2 classes  and max 2 value for features
######Ionosphere Dataset are 2 classes  and max 6 value for features
######ColonTumor Dataset are 2 classes  and max 2 value for features
######ConcentericRectangles Dataset are 3 classes  and max 5 value for features




###### Heart Dataset
Heart=np.loadtxt('Heart.txt')#have 14 columns
df=pd.DataFrame(Heart, columns=["a","b","c","d","e","f","g","h","i","j","k","l","m","target"])
# df_train,df_test=train_test_split(df, test_size=0.30)
# dataset=df_train
# testing_data=df_test
# target=df_train.iloc[:,-1]


######Glass Dataset
# Glass= np.loadtxt('Glass.txt')#have 10 columns
# df=pd.DataFrame(Glass, columns=["a","b","c","d","e","f","g","h","i","target"])
# df_train,df_test=train_test_split(df, test_size=0.30)
# dataset=df_train
# testing_data=df_test
# target=df_train.iloc[:,-1]

#########ConcentericRectangles
# CR= np.loadtxt('Concentric_rectangles.txt')#have 3 columns
# df=pd.DataFrame(CR, columns=["a","b","target"])
# df_train,df_test=train_test_split(df, test_size=0.30)
# dataset=df_train
# testing_data=df_test
# target=df_train.iloc[:,-1]

############Sonar Dataset
# Sonar= np.loadtxt('Sonar.txt')#have 61 columns  with s
# lstSonar=[]
# for i in range(60):
#     lstSonar.append(pd.util.testing.rands(3))
# lstSonar.append("target")
# df=pd.DataFrame(Sonar, columns=lstSonar)

# df_train,df_test=train_test_split(df, test_size=0.30)
# dataset=df_train
# testing_data=df_test
# target=df_train.iloc[:,-1]


##############Ionosphere Dataset
# Ionosphere= np.loadtxt('Ionosphere.txt')#have 35 column with s
# lstIO=[]
# for i in range(34):
#     lstIO.append(pd.util.testing.rands(3))
# lstIO.append("target")
# df=pd.DataFrame(Ionosphere, columns=lstIO)
# df_train,df_test=train_test_split(df, test_size=0.30)
# dataset=df_train
# testing_data=df_test
# target=df_train.iloc[:,-1]


#############ColonTumor Dataset
# ColonTumor= np.loadtxt('Colon Tumor.txt')#have 2001 columns with l
# lstCT=[]
# for i in range(2000):
#     lstCT.append(pd.util.testing.rands(6))
# lstCT.append("target")
# df=pd.DataFrame(ColonTumor, columns=lstCT)
# df_train,df_test=train_test_split(df, test_size=0.30)
# dataset=df_train
# testing_data=df_test
# target=df_train.iloc[:,-1]




###########################################################################################################
###########################################################################################################
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


###########################################################################################################
###########################################################################################################
def InfoGain(data, split_attribute_name, target_name="target"):
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    ##Calculate the entropy of the dataset

    # Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    # Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


###########################################################################################################
###########################################################################################################
def ID3(data, originaldata, features, target_attribute_name="target", parent_node_class=None):
    # Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#

    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    # the mode target feature value is stored in the parent_node_class variable.

    elif len(features) == 0:
        return parent_node_class

    # If none of the above holds true, grow the tree!

    else:
        # Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]


        ###############################################################################################################
        ##########################   sqrt(features) or  log2(features)  ###############################################
        features = np.random.choice(features, size=np.int(np.sqrt(len(features))), replace=False)
        # features = np.random.choice(features, size=np.int(np.log2(len(features))), replace=False)

        # Select the feature which best splits the dataset
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                       features]  # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        # gain in the first run
        tree = {best_feature: {}}

        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]

        # Grow a branch under the root node for each possible value of the root node feature

        for value in np.unique(data[best_feature]):
            value = value
            # Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        return (tree)

    ###########################################################################################################


###########################################################################################################

def predict(query, tree, default=0):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


###########################################################################################################
###########################################################################################################
def train_test_split2(dataset):
    training_data = dataset.iloc[:round(0.75 * len(dataset))].reset_index(
        drop=True)  # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[round(0.75 * len(dataset)):].reset_index(drop=True)


    # training_data, testing_data=train_test_split(dataset, test_size=0.3)
    return training_data, testing_data


# training_data = train_test_split(dataset)[0]
# testing_data = train_test_split(dataset)[1]


###########################################################################################################
###########################################################################################################
#######Train the Random Forest model###########
item_acc = []
for i in range(10):

    df_train, df_test = train_test_split(df, test_size=0.30)
    dataset = df_train
    testing_data = df_test
    target = df_train.iloc[:, -1]
    training_data = train_test_split2(dataset)[0]
    testing_data = train_test_split2(dataset)[1]



    def RandomForest_Train(dataset, number_of_Trees):
        # Create a list in which the single forests are stored
        random_forest_sub_tree = []

        # Create a number of n models
        for i in range(number_of_Trees):
            # Create a number of bootstrap sampled datasets from the original dataset
            bootstrap_sample = dataset.sample(frac=1, replace=True)

            # Create a training and a testing datset by calling the train_test_split function
            bootstrap_training_data = train_test_split(bootstrap_sample)[0]
            bootstrap_testing_data = train_test_split(bootstrap_sample)[1]

            # Grow a tree model for each of the training data
            # We implement the subspace sampling in the ID3 algorithm itself. Hence take a look at the ID3 algorithm above!
            random_forest_sub_tree.append(ID3(bootstrap_training_data, bootstrap_training_data,
                                              bootstrap_training_data.drop(labels=['target'], axis=1).columns))

        return random_forest_sub_tree


    ############## This place we set the number of tree 11 21 31 41 51
    random_forest = RandomForest_Train(dataset, 11)


    #######Predict a new query instance###########
    def RandomForest_Predict(query, random_forest, default=0):
        predictions = []
        for tree in random_forest:
            predictions.append(predict(query, tree, default))
        return sps.mode(predictions)[0][0]


    query = testing_data.iloc[0, :].drop('target').to_dict()
    query_target = testing_data.iloc[0, -1]
    print('target: ', query_target)
    prediction = RandomForest_Predict(query, random_forest)
    print('prediction: ', prediction)




    #######Test the model on the testing data and return the accuracy###########
    def RandomForest_Test(data, random_forest):
        data['predictions'] = None
        for i in range(len(data)):
            query = data.iloc[i, :].drop('target').to_dict()
            data.loc[i, 'predictions'] = RandomForest_Predict(query, random_forest, default=0)
        accuracy = sum(data['predictions'] == data['target']) / len(data) * 100
        # print('The prediction accuracy is: ',sum(data['predictions'] == data['target'])/len(data)*100,'%')

        return accuracy


    item_acc.append(RandomForest_Test(testing_data, random_forest))

# for i in range(10):
#   item_acc.append(RandomForest_Test(testing_data, random_forest))

print(item_acc)

print("The Average of 10 Run Accuracy is: ")
print(np.mean(item_acc))
print("The Std of 10 Run is: ")
print(np.std(item_acc))


