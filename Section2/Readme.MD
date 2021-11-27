
#Dockerfile
Contained custom postgresql image with car_dealership database and also sample data

a) Command to build custom postgresql image
docker build -t ck-postgres-image .

b) Command to run custom docker image with postgresql
docker run -d --name ck-postgres-container -p 5555:5432 ck-postgres-image

#init.sql
Contained sample data needed for car_dealership database

#data dictionary.docx
Contained data dic with ERD

#query.sql
Contained 2 query tasks
