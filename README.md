# My-learning

# Load IRIS Dataset from python inbuilt dataset
iris=datasets.load_iris()

# getting all major info of data
iris.keys()

# looking all aspects of imported data
iris.data
iris.target
iris.target_names
iris.DESCR
iris.feature_names
iris.filename

# Loading arraydata into dataframe for preprocessing and applying ML algo
feats=pd.DataFrame(iris.data,columns=["sepal length","sepal width","petal length","petal width"])

# Looking at Dataframe created
feats

# converting target varibale into Dataframe
response=pd.DataFrame(iris.target,columns=["target"])

# Looking the Dataframe crated for response
response

# Concating both dataset to form a final dataset 
iris=pd.concat([feats,response],axis=1)

# Looking at our final dataset 
iris

# to see first few rows of iris dataset
iris.head(2)

# to see bottom few rows of iris dataset
iris.tail(2)

# Looking at the shape of our data
iris.shape

# Looking at the stats value of data
iris.describe()

# Exploratory Data Analysis through various graphs and plots

# using countplot
fig=plt.figure(figsize=(15,10))
sns.countplot(x="sepal length",hue="target",data=iris)

# using histogram
iris.hist(figsize=(8,10),color="green",edgecolor="black")
plt.show()

# barplot
fig=plt.figure(figsize=(15,7))
sns.barplot(x="petal length",y="target",data=iris)

# scatter plot
sns.scatterplot(x="petal width",y="petal length",data=iris,color="red")

# using boxplot
sns.boxplot(x="sepal width",y="target",data=iris)

# pairplot
sns.pairplot(iris,hue="target")

# using Heatmap,before using heatmap we need to find correlation values of data
iris_corr=iris.corr()
fig=plt.figure(figsize=(10,7))
sns.heatmap(iris_corr,annot=True,cmap="RdYlGn")

# using crosstab
pd.crosstab(index=iris["sepal length"],columns=iris["target"],normalize="index")

# Data preprocessing

# lets check if there is any null value in data
iris.isna().sum()

# as there are no null value we can proceed to model building
# before building model we have to divide our data into dependent and independent variables
y=iris["target"]
x=iris.drop("target",axis=1)

# Before moving to model building we will divide our dataset into train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

# before moving ahead lets check whether we have divided our dataset properly by looking at shape
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Lets check our dataset with Decision tree classifier
dt=DecisionTreeClassifier()
dt=DecisionTreeClassifier(criterion="entropy",max_depth=20)
dt.fit(x_train,y_train)
prediction=dt.predict(x_test)

# lets put the prediction of our model in dataframe for better clarity 
output=pd.DataFrame({"Actual":y_test,"Predicted":prediction})
output

# checking model accuracy 
print("Accuracy of our model is :",accuracy_score(y_test,prediction))
