import numpy as np
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Convert 1D array to a 2D numpy array of 2 rows and 3 columns
arr_2d = np.reshape(arr, (2, 5))
print(arr_2d)


#initialize arrays
A = np.array([[2, 1], [5, 4]])
B = np.array([[3, 4], [7, 8]])

#vertically stack arrays
output = np.vstack((A, B))

print(output)


C = np.array([[2, 1], [5, 4]])
D = np.array([[3, 4], [7, 8]])

#horizontal stack arrays
output = np.hstack((A, B))

print(output)



#How to convert an array of arrays into a flat 1d array?
ini_array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])

# printing initial arrays
print("initial array", str(ini_array1))

result = ini_array1.flatten()

# printing result
print("New resulting array: ", result)









#Convert one dimension to higher dimension
arrn = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
# Convert 1D array to a 2D numpy array of 2 rows and 3 columns
arr_l = np.reshape(arrn, (3, 3))
print(arr_l)





#Create 5x5 an array and find the square of an array?
arr5=np.arange(0,25).reshape(5,5)
arrsq=arr5*arr5
print(arrsq)




#Create 5x6 an array and find the mean

arr6=np.arange(0,30).reshape(5,6)
y=np.mean(arr6)
print(y)




#Find the standard deviation of the previous array
z=np.std(arr6)
print(z)




#Find the median of the previous array in Q8?
w=np.median(arr6)
print(w)




#Find the transpose of the previous array in Q8?
t=np.transpose(arr6)
print(t)




#Create a 4x4 an array and find the sum of diagonal elements?

n_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],[4.0,5.0,9.0], [7.0, 8.0, 9.0]])
print(n_array)


# Finding sum of diagonal elements
outs = np.trace(n_array)
print("Output is: \n", outs)

#Find the determinant of the previous array in Q12?
matrix = np.array([[1, 0, 1], [1, 2, 0], [4, 6, 2]])

print(np.linalg.det(matrix))



#Find the 5th and 95th percentile of an array?
l_array=np.array([5,0,4,1,5,2])
p1=np.percentile(l_array,5)
p2=np.percentile(l_array,95)
print(p1)
print(p2)



#How to find if a given array has any null values

n=l_array[np.where(0)]
if(n.size==0):
    print("no null value")
else:
    print("yes ,it has null values")