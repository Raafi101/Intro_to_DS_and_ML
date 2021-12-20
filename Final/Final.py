# Raafi Rahman
# Stat 72401 Final

# Libraries ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import and clean data ===================================================

# Laptop Path (For my personal machine)
UP_DIR = "C:\\Users\\Rafha\\OneDrive\\Documents\\GitHub\\STAT72401\\Final"

# Desktop Path (For my personal machine)
# UP_DIR = "C:\\Users\\Rafha\\Desktop\\Code\\STAT72401\\Final"

# Open data
csvPath = os.path.join(UP_DIR, 'Houses.csv')
housesRaw = pd.read_csv(csvPath)

# Copy and filter housesRaw and print
housesClean = housesRaw[['property_type', 'purpose', 'bedrooms', 
                        'baths', 'Total_Area', 'city', 'province_name', 
                        'latitude', 'longitude', 'price']].rename(
                        columns = {'Total_Area': 'area', 'property_type': 'type', 'province_name': 'province'})
print(housesClean.head())

# Visualize data ==========================================================

# Target
price = housesClean['price']

# Bedrooms x Price OUTLIERS
housesClean.plot(x = 'bedrooms', y = 'price', kind = 'scatter', alpha = .1)
plt.show()

# Bathrooms x Price OUTLIERS
housesClean.plot(x = 'baths', y = 'price', kind = 'scatter')
plt.show()

# Area x Price OUTLIERS
housesClean.plot(x = 'area', y = 'price', kind = 'scatter')
plt.xlim([0, 99999])
plt.ylim([0, 9999999])
plt.show()
