import numpy as np
import matplotlib.pyplot as plt
from sklearn. linear_model import LinearRegression
#Emample material data
#Feature:
#x1 = atomic number
#x2 = density 
#x3 = formation energy
#y = band ghap (ev)

x =  np.array([
    [14, 2.33, -1.0],
    [22, 4.50, -1.5],
    [32, 5.32, -0.8],
    [42, 6.10, -1.2],
    [48, 7.14, -0.6],
    [13, 2.70, -1.8],
    [31, 5.91, -0.9],
    [16, 2.07, -1.4],
    [34, 4.81, -1.1],
    [8,  1.43, -2.0]
])

y = np.array([
    1.12,
    0.85,
    1.55,
    0.72,
    0.48,
    2.10,
    1.30,
    2.80,
    1.75,
    3.20
])
#create Ml mode
model = LinearRegression()
#Train mode
model.fit(x, y)
#predict band gaps
predicted = model.predict(x)
#print results
print("actual band gap(ev) |predicted band gap (ev)")
for actual, prediction in zip(y, predicted):
    print(f"{actual: .2f} | {prediction: .2f}")
#Model accuracy
r2 = model.score(x, y)
print(r2)
print("\nR2 score:", round(r2, 3))

#plot actual vs predicted 
plt . scatter(y, predicted)
plt.xlabel("Actual Band Gap (ev)")
plt.ylabel("predicted Band Gap (ev)")
plt.title("Material Band Gaph prediction using Machine Learning")
#perfect prediction line
plt.plot([min(y), max(y)], [min(y), max(y)])
plt.show()