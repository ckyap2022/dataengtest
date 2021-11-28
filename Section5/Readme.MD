#Section 5: Machine Learning Using the dataset from https://archive.ics.uci.edu/ml/datasets/Car+Evaluation, 
create a machine learning model to predict the buying price given the following parameters:

a) Maintenance = High 

b) Number of doors = 4 

c) Lug Boot Size = Big 

d) Safety = High 

e) Class Value = Good

#Attribute Values:

buying       v-high, high, med, low

maint        v-high, high, med, low

doors        2, 3, 4, 5-more

persons      2, 4, more

lug_boot     small, med, big

safety       low, med, high

class values unacc, acc, good, vgood


#Sample Data (car.data file):

vhigh,vhigh,2,2,small,low,unacc
vhigh,vhigh,2,2,small,med,unacc
vhigh,vhigh,2,2,small,high,unacc
vhigh,vhigh,2,2,med,low,unacc
vhigh,vhigh,2,2,med,med,unacc
vhigh,vhigh,2,2,med,high,unacc
vhigh,vhigh,2,2,big,low,unacc
vhigh,vhigh,2,2,big,med,unacc
vhigh,vhigh,2,2,big,high,unacc
vhigh,vhigh,2,4,small,low,unacc
vhigh,vhigh,2,4,small,med,unacc

#The model evaluates cars according to the following concept structure:

   CAR                      car acceptability

. PRICE                  overall price

. . buying               buying price

. . maint                price of the maintenance

. TECH                   technical characteristics

. . COMFORT              comfort

. . . doors              number of doors

. . . persons            capacity in terms of persons to carry

. . . lug_boot           the size of luggage boot

. . safety               estimated safety of the car


#Car price evaluation is uploaded to the repos and the file name is car_evaluation.py

#Car sample data called car_data.csv is uploaded to repos.

#Output from Decision Tree Regression Model

Predicted price: for maintainance =  72

Predicted price: for doors =  108

Predicted price: for lug_boot =  90

Predicted price: for safety =  90

Predicted price: for class =  222
