import cmath
import math
import random



#Import the numpy package under the name np
import numpy as np
#Create a null vector of size 10
arr1=np.array([0]*10)
print(arr1)
#Create a vector with values ranging from 10 to 49
arr2=np.arange(10, 50)
print(arr2)

#Find the shape of previous array in question 3
print(arr2.shape)

#Print the type of the previous array in question 3
print(arr2.dtype)

#Print the numpy version and the configuration
print(np.__version__)
print(np.show_config())

#Print the dimension of the array in question 3
print(arr2.ndim)

#Create a boolean array with all the True value
arr3=np.array([True]*5)
print(arr3)

#Create a two dimensional array
arr4=np.array([[2,1,0],[5,1,6]])
print(arr4)


#Create a three dimensional array
arr5=np.array([[[5,2,1],[5,2,4],[41,7,8]]])
print(arr5)


#-----------------------
#Difficulty Level Easy

#Reverse a vector (first element becomes last)
arr6=np.arange(0,10)
print(arr6)
print(arr6[::-1])

#Create a null vector of size 10 but the fifth value which is 1
arr7=np.array([int(x==4) for x in range(10)])
print(arr7)



#Create a 3x3 identity matrix
obj1 = np.identity(3, dtype = int)
print(obj1)


#Convert the data type of the given array from int to float
arr8 = np.array([1, 2, 3, 4, 5])
arr9=arr8.astype(np.float64)
print(arr9.dtype)


arr10 = np.array([[1., 2., 3.],

            [4., 5., 6.]])

arr11 = np.array([[0., 4., 1.],
 [7., 2., 12.]])
#Multiply arr1 with arr2
arr12=arr10*arr11
print(arr12)


arr14 = np.array([[1., 2., 3.],

            [4., 5., 6.]])

arr15 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
#Make an array by comparing both the arrays provided above
arr16=arr14-arr15
print(arr16)

print((arr14==arr15).all())
#Extract all odd numbers from arr with values(0-9)
print("-------------")
arrodd=np.arange(0,10)
j=np.where(arrodd%2!=0)
print(j)
print("*******")
#Replace all odd numbers to -1 from previous array
arrodd=np.arange(0,10)
arrodd[np.where(arrodd%2!=0)]=1

print(arrodd)

#arr = np.arange(10)
#Replace the values of indexes 5,6,7 and 8 to 12

arr18=np.arange(0,10)
arr18[5:9]=12
print(arr18)

#Create a 2d array with 1 on the border and 0 inside
arr19=np.array([[4,0,0,4],[5,0,0,5]])
print(arr19)
arr19[0:2,0:1]=1
arr19[0:2,3:4]=1
print(arr19)


#Difficulty Level Medium


arr2d = np.array([[1, 2, 3],

            [4, 5, 6],

            [7, 8, 9]])
#Replace the value 5 to 12
arr2d[1,1]=12
print(arr2d)


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
#Convert all the values of 1st array to 64
arr3d[1,]=64
print(arr3d)


#Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it
arr21=np.arange(0,10).reshape(2,5)
print(arr21)
print(arr21[0,])
#Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it
arr22=np.arange(10,20).reshape(2,5)
print(arr22[1,1])
#Make a 2 - Dimensional array with values 0-9 and slice out the third column but only the first two rows
arr22=np.arange(10,20).reshape(2,5)
print(arr22[0:2,2])

#Createa10x10arraywith random values and find the minimum and maximum values

arr24= np.random.random((10,10))
print("Original Array:")
print(arr24)
xmin, xmax = arr24.min(), arr24.max()
print("Minimum and Maximum Values:")
print(xmin, xmax)

#27
arr25 = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
arr26 = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
#Findthecommonitemsbetweena and b

print(np.intersect1d(arr25, arr26))
list1=[]
for i in range(0,10):
     for j in range(0,10):

        if (arr25[i]==arr26[j]):
            list1.append(arr25[i])
            break
if(len(list1)==0):
    print("not found")
print(list(set(list1)))


c = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
d= np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
list2=[]
list3=[]
for i in range(0,10):
     for j in range(0,10):

        if (c[i]==d[j]):
            list2.append(i)
            list3.append(j)
            break
if(len(list2) ==0):
    print("not found")
print(list2,list3)

e = np.array([1,2,3,2,3,4,3,4,5,6])
f = np.array([7,2,10,2,7,4,9,4,9,8])
#Find the positions where elements of a and b match
c = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
d= np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
list2=[]
list3=[]
for i in range(0,10):
     for j in range(0,10):

        if (c[i]==d[j]):
            list2.append(i)
            list3.append(j)
            break
if(len(list2) ==0):
    print("not found")
print(list2,list3)



names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randint(7,size=random.randrange(1,10))
print(data)
print(np.where(names[data]!="Will"))
#Find all the values from array data where the values from array names are not equal to Will



names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

#Find all the values from array data where the values from array names are not equal to Will and Joe
data = np.random.randint(7,size=random.randrange(1,10))
print(data)
print(np.where((names[data]!="Will") & (names[data]!="Joe")))




#Difficulty Level Hard
#Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.
m = np.arange(0, 20).reshape(5, 4)
print(m)
arr30=np.arange(1,16).reshape(5,3)
print(arr30)
arr40=np.linspace(1,15,15).reshape(5,3)
print(arr40)


#Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.
arr31=np.random.uniform(low=1.0,high=15.0,size=(5,3))
print(arr31)
#Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.
arr32=np.random.uniform(low=1.0,high=15.0,size=(2,2,4))
print(arr32)

#Swap axes of the array you created in Question 32
np.swapaxes(arr31,0,1)
print(arr31)


#Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0
arr33=np.array(range(10))
print(arr33*arr33)
for i in range(0,10):
    if(arr33[i]<0.5):
        arr33[i]=0
print(arr33)


#Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

arr50=np.random.randint(12,size=(12))
arr51=np.random.randint(12,size=(12))

print(arr50)
print(arr51)


arr52=np.maximum(arr50, arr51)
print(arr52)


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
#Find the unique names and sort them out!
arr44=np.unique(names)
print(arr44)






arr40 = np.array([1,2,3,4,5])
arr41 = np.array([5,6,7,8,9])
#From array a remove all items present in array b

arr42=np.intersect1d(arr41,arr40)
print(arr42)
arr43 = np.delete(arr40,4 )

arr40=arr43
print(arr40)
#Following is the input NumPy array delete column two and insert following new column in its place.
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
print(sampleArray[:3,1:2])
hello=np.swapaxes(newColumn,0,1)

#arrnew = np.delete(sampleArray, 1, 1)
sampleArray[:3,1:2]=hello
print(sampleArray)


x1 = np.array([[1., 2., 3.], [4., 5., 6.]])
y2 = np.array([[6., 23.], [-1, 7], [8, 9]])

g = x1.dot(y2)

print(g)


#Generate a matrix of 20 random values and find its cumulative sum
arrlast=np.random.randint(0,20,20,int)
print(np.cumsum(arrlast))
