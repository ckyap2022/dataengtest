Readme.MD

#Input
Assumed user will upload the files (dataset1.csv and dataset2.csv) manually to c:\gov

#Process
The py job will process all the csv files uploaded into c:\gov based on the following processing tasks :

a) Split the name field into first_name, and last_name
b) Remove any zeros prepended to the price field
c) Delete any rows which do not have a name
d) Create a new field named above_100, which is true if the price is strictly greater than 100

#Scheduler
a) Schedule using Apache airflow to run 1am everyday
b) The processed datasets will be stored in c:\gov\output
