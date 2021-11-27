import pandas as pd
import glob
import os, errno

# Getting csv files from the folder gov
path = "C:\\gov\\"
output_path = path + "\\output"

# Read all the files with extension .csv
filenames = glob.glob(path + "\*.csv")
print('File names:', filenames)

# Create output folder for storing processed datasets
try:
	os.makedirs(output_path)
except OSError as e:
   	if e.errno != e.errno != errno.EEXIST:
   		raise

# for loop to iterate all csv files
for file in filenames:
   # reading csv files
   print("\nReading file = ",file)
   df = pd.read_csv(file)

   # Delete any rows which do not have a name
   df.dropna(subset = ['name'], inplace = True) 

   # Split the name field into first_name, and last_name
   df['first name'] = df['name'].astype(str).str.split(' ', n=1, expand=True).get(0)
   df['last name'] = df['name'].astype(str).str.split(' ', n=1, expand=True).get(1)

   try: 
   	# Remove any zeros prepended to the price field
   	df['price'] = df['price'].astype(str).str.lstrip('0')

   	# Create a new field named above_100, which is true if the price is strictly greater than 100
   	df.loc[df['price'].astype(float) <= 100, 'above_100'] = 'false' 
   	df.loc[df['price'].astype(float) > 100, 'above_100'] = 'true' 
   except:
    	df['above_100'] = 'false'

   # Write processed dataset
   df.to_csv(output_path + "\\out_"+os.path.basename(file))
   del df
   
   







   

