# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

### **1. Collect the Data**

Obtain customer data relevant to segmentation. Common features include:

* Age
* Gender
* Annual Income
* Spending Score
* Purchase History

### **2. Preprocess the Data**

Prepare the data by:

* Removing missing or irrelevant values
* Converting categorical variables to numerical if needed
* Scaling/normalizing the features to ensure equal importance (especially for distance-based methods like K-Means)
### **3. Select Features for Clustering**

Choose the features that will help you identify meaningful customer segments. For example:

* Age and Spending Score
* Income and Purchase Frequency
  Select 2 or 3 to make visualization easier.

### **4. Determine the Number of Clusters (k)**

Use methods like:

* **Elbow Method**: Plot within-cluster sum of squares (WCSS) against number of clusters and look for an “elbow” point.
* **Silhouette Score**: Measures how similar a point is to its own cluster compared to other clusters.

### **5. Apply K-Means Clustering**

Use the selected number of clusters (k) and apply the K-Means algorithm to:

* Randomly initialize centroids
* Assign each data point to the nearest centroid
* Recalculate centroids based on assignments
* Repeat until convergence (no change in assignments or centroids)

### **6. Assign Cluster Labels**

After the algorithm finishes, each customer is assigned to a cluster. These clusters represent different segments.

### **7. Visualize the Clusters**

Plot the clusters using scatter plots (if using 2 or 3 features) to visually understand the separation and characteristics of each segment.


### **8. Interpret the Segments**

Analyze the average values of features within each cluster to describe the segments. For example:

* Cluster 0: Young, high spending customers
* Cluster 1: Older, low spending customers
* etc.

### **9. Take Action Based on Segments**

Use the insights to:

* Personalize marketing strategies
* Design targeted promotions
* Improve customer retention

 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Reshma C
RegisterNumber:  212223040168
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
# DATA.HEAD():
![image](https://github.com/user-attachments/assets/3fcb7fdd-f5fc-4b0c-8ab7-4dd9e357c0d0)

# DATA.INF0():
![image](https://github.com/user-attachments/assets/6d69fcd4-3902-4de0-b8c5-d7e7ea3f7760)


# DATA.ISNULL().SUM():

![image](https://github.com/user-attachments/assets/9cf2ea2a-c736-4540-abeb-cf94c53aeeff)

# PLOT USING ELBOW METHOD:


![image](https://github.com/user-attachments/assets/2e9b19a1-8d75-42f3-9790-483a57b54b42)


# CUSTOMER SEGMENT:

![image](https://github.com/user-attachments/assets/cbb82036-b566-473f-85de-cb3baa2e5668)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
