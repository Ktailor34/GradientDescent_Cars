# Kishan Tailor
# 10/18/2020
# Project 2: Computing Linear Regressions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# given by proffesor for R^2 value
def r2_score(Y, Y_pred):
  
  mean_y = np.mean(Y)
  ss_tot = sum((Y - mean_y) ** 2)
  ss_res = sum((Y - Y_pred) ** 2)
  r2 = 1 - (ss_res / ss_tot)
  return r2

# def plotgraph(currentx, mpg):
#   currenty = mpg
#   fig, ax = plt.scatter(currentx,currenty)

# given file
data = pd.read_csv('auto-mpg.csv')

# separate the values from each column by column name
mpg = data['mpg'].values
displacement = data['displacement'].values #continous 
horsepower = data['horsepower'].values #continous 
weight = data['weight'].values #continous 
acceleration = data['acceleration'].values #continous 
cylinders = data['cylinders'].values #categorical
year = data['year'].values #categorical
origin = data['origin'].values #categorical
#car = data['car'].values #String?

# generate a matrix of ones for x0
n = float(len(displacement))

# create an array of all features , i.e. X values , and transpose it
Xs = np.array ([displacement, horsepower, weight, acceleration]).T

# create an array for output variables
Y = np.array(mpg)

# establish a learning rate
alpha = 0.0000001

c = 0
m = np.zeros(4)

errorList = []
r2List = []
# Find m and c for the non-normalized data using least squares regression

# scaler = StandardScaler().fit(Xs)
# Xs = scaler.transform(Xs)
print()
currenterror = 999999999999999999999999999999
for i in range(500000):
  Y_pred = np.dot(m,Xs.T) + c  # The current predicted value of Y

  error = (1/n) * np.sum(np.square(Y_pred - Y))

  if (error > currenterror):
    break
  currenterror = error

  D_m1 = (-2/n) * np.sum(Xs.T[0] * (Y-Y_pred))  # Derivative wrt m
  D_m2 = (-2/n) * np.sum(Xs.T[1] * (Y-Y_pred))  # Derivative wrt m
  D_m3 = (-2/n) * np.sum(Xs.T[2] * (Y-Y_pred))  # Derivative wrt m
  D_m4 = (-2/n) * np.sum(Xs.T[3] * (Y-Y_pred))  # Derivative wrt m
  # D_m5 = (-2/n) * np.sum(Xs.T[4] * (Y-Y_pred))  # Derivative wrt m
  # D_m6 = (-2/n) * np.sum(Xs.T[5] * (Y-Y_pred))  # Derivative wrt m
  # D_m7 = (-2/n) * np.sum(Xs.T[6] * (Y-Y_pred))  # Derivative wrt m
  D_c = (-2/n) * np.sum(Y-Y_pred)  # Derivative wrt c
  m[0] = m[0] - (alpha * D_m1)  # Update m
  m[1] = m[1] - (alpha * D_m2)  # Update m
  m[2] = m[2] - (alpha * D_m3)  # Update m
  m[3] = m[3] - (alpha * D_m4)  # Update m
  # m[4] = m[4] - (alpha * D_m5)  # Update m
  # m[5] = m[5] - (alpha * D_m6)  # Update m
  # m[6] = m[6] - (alpha * D_m7)  # Update m
  c = c - (alpha * D_c)  # Update c
  
  errorList.append(error)
  r2List.append(r2_score(Y,Y_pred))
  print(error)


print ("error:", error)
print("r2",r2_score(Y,Y_pred))
print(m,c)


#### Find all plotting below

# iterationDomain = [i for i in range (500000)]

#plt.plot(iterationDomain, errorList)
# plt.xlabel("iterations")
# plt.ylabel("cost")
# plt.title("iterations vs cost")
# plt.savefig("un-normalized iterations vs cost")

# plt.plot(iterationDomain, r2List)
# plt.xlabel("iterations")
# plt.ylabel("r^2 values") 
# plt.title("iterations vs r^2")
# plt.savefig("un-normalized iterations vs r^2")

#plt.plot([i for i in range (20000)], errorList)
#plt.plot([i for i in range (20000)], r2List)

# plt.scatter(displacement, mpg)
# plt.xlabel("displacement")
# plt.ylabel("mpg")
# plt.title("displacement vs mpg")

# plt.scatter(horsepower, mpg)
# plt.xlabel("horsepower")
# plt.ylabel("mpg")
# plt.title("horsepower vs mpg")

# plt.scatter(weight, mpg)
# plt.xlabel("weight")
# plt.ylabel("mpg")
# plt.title("weight vs mpg")

# plt.scatter(acceleration, mpg)
# plt.xlabel("acceleration")
# plt.ylabel("mpg")
# plt.title("acceleration vs mpg")

# plt.scatter(cylinders, mpg)
# plt.xlabel("cylinders")
# plt.ylabel("mpg")
# plt.title("cylinders vs mpg")

# plt.scatter(year, mpg)
# plt.xlabel("year")
# plt.ylabel("mpg")
# plt.title("year vs mpg")

# plt.scatter(origin, mpg)
# plt.xlabel("origin")
# plt.ylabel("mpg")
# plt.title("origin vs mpg")

#plt.savefig("displacement vs mpg")
# plt.show()
  

