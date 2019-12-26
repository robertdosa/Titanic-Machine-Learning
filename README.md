# Titanic-Machine-Learning
Applying different machine learning algorithms on the famous Titanic dataset

Source of the dataset: https://www.kaggle.com/c/titanic/data


The purpose of this repository to demonstrate different classification algorithms on the same dataset. Since it is a well-known dataset I did not made any exploratory data analysis. Different notebooks will be add in the future.


Let's have a look at the dataset.


<table border="1">
<thead>
<tr>
<th>Pass.Id</th>
<th>Surv.</th>
<th>Pclass</th>
<th>Name</th>
<th>Sex</th>
<th>Age</th>
<th>SibSp</th>
<th>Parch</th>
<th>Ticket</th>
<th>Fare</th>
<th>Cabin</th>
<th>Emb.</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>0</td>
<td>3</td>
<td>Braund, Mr. Owen Harris</td>
<td>male</td>
<td>22.0</td>
<td>1</td>
<td>0</td>
<td>A/5 21171</td>
<td>7.2500</td>
<td>NaN</td>
<td>S</td>
</tr>
<tr>
<td>2</td>
<td>1</td>
<td>1</td>
<td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
<td>female</td>
<td>38.0</td>
<td>1</td>
<td>0</td>
<td>PC 17599</td>
<td>71.2833</td>
<td>C85</td>
<td>C</td>
</tr>
<tr>
<td>3</td>
<td>1</td>
<td>3</td>
<td>Heikkinen, Miss. Laina</td>
<td>female</td>
<td>26.0</td>
<td>0</td>
<td>0</td>
<td>STON/O2. 3101282</td>
<td>7.9250</td>
<td>NaN</td>
<td>S</td>
</tr>
<tr>
<td>4</td>
<td>1</td>
<td>1</td>
<td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
<td>female</td>
<td>35.0</td>
<td>1</td>
<td>0</td>
<td>113803</td>
<td>53.1000</td>
<td>C123</td>
<td>S</td>
</tr>
<tr>
<td>5</td>
<td>0</td>
<td>3</td>
<td>Allen, Mr. William Henry</td>
<td>male</td>
<td>35.0</td>
<td>0</td>
<td>0</td>
<td>373450</td>
<td>8.0500</td>
<td>NaN</td>
<td>S</td>
</tr>
</tbody>
</table>
<p>'</p>


Variable Notes (Source: Kaggle)

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.


History:

2019.12.22.: Random forest classification : Titanic_random_forest_classification.ipynb
2019.12.26.: K-nearest neighbors classification: Titanic_K-nearest_neighbors_classification.ipynb
