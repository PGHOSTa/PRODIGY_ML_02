# PRODIGY_ML_02

```markdown
# Mall Customer Segmentation using K-Means Clustering

This project applies K-Means clustering to a dataset of mall customers to identify distinct customer segments based on their numerical features. The clustering results are then visualized and saved to a new CSV file with the corresponding cluster labels.

## Requirements

Ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset

This project uses the dataset `Mall_Customers.csv` which contains information about customers such as their age, annual income, and spending score. Ensure the dataset is available in the directory or update the file path in the code to the correct location.

##Dataset Overview:

![k-means 4](https://github.com/user-attachments/assets/3788483c-0acf-41e9-ab05-81f24c07aecc)

## Steps

### 1. Load and Explore the Dataset
The dataset is loaded using the `pandas` library, and the first few rows of data are displayed along with the columns.

### 2. Visualize Pairwise Relationships
A pairplot is generated using `seaborn` to visualize the relationships between the numerical features and detect potential patterns.

![k-means 1](https://github.com/user-attachments/assets/39325807-958c-46ab-a0dd-bf7049b571b6)


### 3. Preprocess the Data
The numerical features are selected, and the data is standardized using `StandardScaler` from `sklearn` to ensure that all features are on the same scale.

### 4. Apply K-Means Clustering
The optimal number of clusters is determined using the elbow method. The inertia (sum of squared distances of samples to their closest cluster center) is plotted against the number of clusters, and the "elbow" point indicates the ideal `k`.

### 5. Visualize the Clusters
If there are at least two numerical features, the clusters are visualized in a scatter plot with the first two features. The centroids of the clusters are marked with red "X" markers.

![k-means 3](https://github.com/user-attachments/assets/6d7e4e3e-d1a9-480e-ba1f-0a7652fdf24c)



### 6. Save the Results
The dataset with the predicted cluster labels is saved to a new CSV file (`clustered_data.csv`).

## Instructions to Run the Code

1. **Download the dataset**: Ensure that the `Mall_Customers.csv` file is available at the specified path or modify the `file_path` variable to point to the correct location.

2. **Run the code**: Execute the Python script. The elbow plot will be displayed, and you will be prompted to input the optimal number of clusters (`k`) based on the elbow plot.

3. **View the results**: After completing the clustering process, the dataset with cluster labels will be saved to a new CSV file named `clustered_data.csv`.

## Example Elbow Plot

The elbow plot helps in determining the optimal number of clusters by plotting the inertia for different values of `k`. The optimal value of `k` is typically chosen where the inertia begins to level off.

![k-means 2](https://github.com/user-attachments/assets/c67b9ad1-40bc-4e05-84f9-bada60cace57)

![k-means 5](https://github.com/user-attachments/assets/0253bce9-2620-4344-98e3-a2c3a90717a0)


## Conclusion

This project demonstrates the use of K-Means clustering to segment customers into distinct groups based on their features. This type of customer segmentation can be useful for targeted marketing, product recommendations, or business strategy.

---

## Notes

- If the dataset has fewer than two numerical features, the 2D visualization will not be possible.
- The code asks for manual input to specify the optimal number of clusters based on the elbow plot.

