#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For data handling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Linear Regression example

# https://scikit-learn.org/stable/datasets/index.html#diabetes-dataset

# In[2]:


# Load the dataset
from sklearn import datasets
diabetes = datasets.load_diabetes()


# In[3]:


# Get input, output, and their names
X = diabetes.data
y = diabetes.target
names = diabetes.feature_names


# In[4]:


# First 5 data points
df = pd.DataFrame(np.concatenate((X, y[:, np.newaxis]), axis=1), columns=names + ['disease progression'])
df.head()


# In[5]:


# Only one feature
bmi = X[:, np.newaxis, 2]


# In[6]:


# Plot bmi vs. target variable
plt.figure(dpi=200)
plt.scatter(bmi, y)
plt.xlabel('bmi')
plt.ylabel('disease progression')


# In[7]:


# Split the dataset into train and test

X_train = bmi[:300]
y_train = y[:300]

X_test = bmi[300:]
y_test = y[300:]


# In[8]:


# Sanity check the shapes
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[9]:


# Plot train and test
plt.figure(dpi=200)
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.xlabel('bmi')
plt.ylabel('disease progression')
plt.legend()


# In[10]:


# Call the linear regression object
from sklearn import linear_model
model1 = linear_model.LinearRegression()


# In[11]:


# LEARN = .fit(input, target)
model1.fit(X_train, y_train)


# In[12]:


# Get predictions for the test set
preds = model1.predict(X_test)


# In[13]:


# Check predictions vs. labels for the first 10 data points
preds[:10], y_test[:10]


# In[14]:


# Quantify the error
preds[:10] - y_test[:10]


# In[15]:


# Use absolute values to avoid positive/negative errors canceling each other
abs(preds[:10] - y_test[:10])


# In[16]:


# Find the mean abs error
np.mean(abs(preds[:10] - y_test[:10]))


# In[17]:


# Find the mean abs error, for all test data
np.mean(abs(preds - y_test))


# In[18]:


# This and many other metrics are available in sklearn
from sklearn import metrics
metrics.mean_absolute_error(y_test, preds)


# In[19]:


# Plot train/test data with fit
plt.figure(dpi=200)
plt.scatter(X_train, y_train, label='train')
plt.plot(X_test, model1.predict(X_test), label='fit')
plt.scatter(X_test, y_test, label='test')
plt.xlabel('bmi')
plt.ylabel('disease progression')
plt.legend()


# # Deep learning example

# In[20]:


# Download the MNIST ataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[21]:


# Random digit from the training set.
index = np.random.randint(0, 60000)
plt.imshow(x_train[index])
print('Label: ', y_train[index])


# In[22]:


# Build the architecture
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Activation, Dense

model2 = Sequential()
model2.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
model2.add(Activation('relu'))
model2.add(Conv2D(32, (3, 3), strides=2, padding='same'))
model2.add(Activation('relu'))
model2.add(Flatten())
model2.add(Dense(10))
model2.add(Activation('softmax'))
model2.summary()


# In[23]:


# Set the optimizer and the loss
from keras.optimizers import Adam, SGD
opt = SGD(lr=0.001)
model2.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# In[24]:


# Put data into suitable shape
from keras.utils import to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# In[25]:


# Train and test
H = model2.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test))


# In[26]:


# Random image from the test set
index = np.random.randint(0, 10000)
random_image = x_test[np.newaxis, index]
plt.imshow(random_image.reshape(28, 28))
print('Label:', np.argmax(y_test[index]))
print('Prediction: ', np.argmax(model2.predict(random_image)))  # model.predict(image)


# # Challenge 1

# In[27]:


from sklearn import datasets
diabetes = datasets.load_diabetes()


# - Repeat the example: predict disease progression from bmi
# 
# - Check errors with different metrics e.g. mean squared error (mse)
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# 
# - Compare mse with mae
# 
# - Find the intercept and slope using sklearn.linear_model.LinearRegression attributes
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# 
# - Instead of bmi only, use more features and fit again, check if the performance improves
# 
# - Use other algorithms e.g. Random Forests

# # Data = Signal + Noise

# - Data generation
#     - Signal
#     - Noise
#     - Signal + Noise

# In[28]:


# Newton's second law as signal

m = 2  # kg
F = 10  # N

a = F/m  # m/s2

# a = intercept + slope * F

a


# In[29]:


# 10 random F's between 0 and 1
F = np.random.rand(10, 1)
F


# In[30]:


# 10 random F's between 0 and 100
F = np.random.rand(10, 1)*100
F


# In[31]:


# Calculate a using Newton's second law
a = F/m
a


# In[32]:


# Plot F vs. a
plt.figure(dpi=200)
plt.scatter(F, a)
plt.xlabel('F (N)')
plt.ylabel('a (m/s2)')


# In[33]:


# Let's add noise to the signal
m = 2  # kg
F = np.random.rand(10, 1) * 100  # N
noise = np.random.rand(10, 1) * 10 - 5
a = F/m + noise  # m/s2

# Plot F vs. a
plt.figure(dpi=200)
plt.scatter(F, a)
plt.xlabel('F (N)')
plt.ylabel('a (m/s2)')


# **Data = Signal + Noise**
# 
# a = F/m + noise

# In[34]:


# Signal vs. Data
plt.figure(dpi=200)
plt.scatter(F, a, label='data, F/m + noise')
plt.scatter(F, F/m, label='signal, F/m')
plt.xlabel('F (N)')
plt.ylabel('a (m/s2)')
plt.legend()


# In[35]:


# Data = Signal + Noise
plt.figure(dpi=200)
plt.scatter(F, a, label='data, F/m + noise')
plt.plot(F, F/m, label='signal, F/m', color='orange')

for i in range(len(F)):
    plt.plot([F[i], F[i]], [F[i]/m, a[i]], 'k--', label='noise' if i == 0 else None)

plt.xlabel('F (N)')
plt.ylabel('a (m/s2)')
plt.legend()
plt.title('Data = Signal + Noise')


# **The aim of machine learning is to find the signal**

# In[36]:


# We know the signal, let's see if ML can find it
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(F, a)


# In[37]:


# Signal vs. Fit
plt.figure(dpi=200)
plt.scatter(F, a, label='data')
plt.plot(F, F/m, label='signal', color='orange')
plt.plot(F, linear_model.predict(F), label='fit')

for i in range(len(F)):
    plt.plot([F[i], F[i]], [F[i]/m, a[i]], 'k--', label='noise' if i == 0 else None)

plt.xlabel('F (N)')
plt.ylabel('a (m/s2)')
plt.legend()


# In[38]:


# Aleatoric vs. Epistemic uncertainty
plt.figure(dpi=200)
plt.plot([0, 100], [-10, 60], label='fit')
plt.plot([0, 100], [0, 50], label='signal')
plt.scatter([20], [20], label='data')
plt.plot([20, 20], [4, 10], '--', label='epistemic uncertainty')
plt.plot([20, 20], [10, 20], '--', label='aleatoric uncertainty = noise')
plt.legend()


# # Challenge 2

# - Plot how epistemic and aleatoric uncertainity will change wrt the number of data. (fixed noise)
# - Plot how epistemic and aleatoric uncertainity will change wrt the noise. (fixed number of data)
# - Make an educated guess before plotting.

# # Overfitting

# In[39]:


# Plot again the prev. data
plt.figure(dpi=200)
plt.scatter(F, a, label='data')
plt.plot(F, F/m, label='signal', color='orange')
plt.plot(F, linear_model.predict(F), label='fit')

plt.xlabel('F (N)')
plt.ylabel('a (m/s2)')
plt.legend()


# In[40]:


# Fit a degree 5 polynomial to F and a
z = np.poly1d(np.polyfit(F.reshape(-1), a.reshape(-1), 5))


# In[41]:


# Plot both fits
plt.figure(dpi=200)
plt.scatter(F, a, label='data')
plt.plot(F, F/m, label='signal', color='orange')
plt.plot(F, linear_model.predict(F), label='degree 1')

plt.plot(np.arange(min(F), max(F)), z(np.arange(min(F), max(F))), label='degree 5', color='red')

plt.xlabel('F (N)')
plt.ylabel('a (m/s2)')
plt.legend()


# In[42]:


# Generate new data
m = 2  # kg
F_new = np.random.rand(10, 1) * (max(F)-min(F)) + min(F)  # N
noise = np.random.rand(10, 1) * 10 - 5
a_new = F_new/m + noise  # m/s2


# In[43]:


# Both fits on train and test data
plt.figure(dpi=200)
plt.scatter(F, a, label='data')
plt.scatter(F_new, a_new, label='new data')
plt.plot(F, F/m, label='signal', color='orange')
plt.plot(F, linear_model.predict(F), label='degree 1')

plt.plot(np.arange(min(F), max(F)), z(np.arange(min(F), max(F))), label='degree 5', color='red')

plt.xlabel('F (N)')
plt.ylabel('a (m/s2)')
plt.legend()


# # Challenge 3

# - Calculate training error for both fits
# - Calculate test error for both fits
# - Fit different degree polynomials and plot training and test errors wrt degree of the polynomial fit
# - Generate data using a quartic signal, repeat previous step, and discover yourself what "underfitting" is.

# # Challenge 4 

# Implement linear regression using gradient descent

# In[44]:


import numpy as np
import matplotlib.pyplot as plt


# In[45]:


# Generate data
# 1. Define m and n
m = 10  # number of examples
n = 1  # number of features


# In[46]:


# 2. Generate X
X = np.random.uniform(size=(m, n))
X


# In[47]:


# 3. Add 1's to X for the bias term for the dot product
X = np.concatenate((np.ones((m, 1)), X), axis=1)
X


# In[48]:


# 4. Randomly generate parameters
theta_true = np.random.uniform(size=(n+1, 1))
theta_true


# In[49]:


# 5. Calculate y
y = np.dot(X, theta_true)
y


# In[50]:


# 6. Plot the generated X and y
plt.figure(dpi=200)
plt.scatter(X[:, 1], y)


# In[51]:


def plotter(X, y, theta):
    '''
    Arguments:
        n = 1
        X: np array with shape (m, 1+n)
        y: np array with shape (m, 1)
        theta: np array with shape (1+n, 1)
    Returns:
        None
        Plots X, y and the fit defined by theta
    '''
    plt.figure(dpi=200)
    plt.scatter(X[:, 1], y)
    linex = np.linspace(X[:, 1].min(), X[:, 1].max())
    liney = theta[0] + theta[1]*linex
    plt.plot(linex, liney)


# In[52]:


# 7. Plot the generated X and y, together with the signal used to generate them
plotter(X, y, theta_true)


# ## Gradient Descent

# In[53]:


# Random weight initialization
theta = np.random.uniform(size=(n+1, 1))
theta


# In[54]:


# X, y together with random fit
plotter(X, y, theta)


# In[55]:


# Sub-challenge 1
def cost_function(X, y, theta):
    '''
    Arguments:
        X: np array with shape (m, 1+n)
        y: np array with shape (m, 1)
        theta: np array with shape (1+n, 1)
    Returns:
        cost, a float.
    '''
    
    return cost


# In[56]:


# It should be a positive number
cost_function(X, y, theta)


# In[57]:


# It should be zero
cost_function(X, y, theta_true)


# In[58]:


# Sub-challenge 2
# Take the theta, update it using the GD rule
def GD_one_step(X, y, theta, lr):
    '''
    Updates theta using one gradient descent step.
    Arguments:
        X: np array with shape (m, 1+n)
        y: np array with shape (m, 1)
        theta: np array with shape (1+n, 1)
        lr: learning rate, a float
    Returns:
        theta_new: np array with shape (1+n, 1), updated after one GD step
    '''
    
    return theta_new


# In[59]:


print('theta: ', theta.T)  # Theta before update
print('theta_true: ', theta_true.T)  # True theta


# In[60]:


theta = GD_one_step(X, y, theta, lr=0.01)


# In[61]:


print('theta: ', theta.T) # Theta after the first update
print('theta_true: ', theta_true.T)  # True theta


# In[62]:


# Sub-challenge 3
# Repeat update
def GD(X, y, lr, epoch):
    '''
    Finds theta from X and y using gradient descent.
    Starts by random init. of theta
    Repeats GD_one_step for predefined number of epochs
    Arguments:
        X: np array with shape (m, 1+n)
        y: np array with shape (m, 1)
        lr: learning rate, a float
        epoch: an integer
    Returns:
        theta: np array with shape (1+n, 1), final theta after training
    '''
    
    return theta


# In[63]:


# Linear regression with Gradient Descent
theta = GD(X, y, lr=0.01, epoch=10000)


# In[64]:


# Plot the learned theta
plotter(X, y, theta)


# In[65]:


def plotter_multiple(X, y, theta, theta_true):
    '''
    For comparing the learned theta with the true theta
    Arguments:
        n = 1
        X: np array with shape (m, 1+n)
        y: np array with shape (m, 1)
        theta: np array with shape (1+n, 1)
        theta_true: np array with shape (1+n, 1)
    Returns:
        None
        Plots X, y, fit(theta), signal(theta_true)
    '''
    plt.figure(dpi=200)
    plt.scatter(X[:, 1], y)
    line_x = np.array([X[:, 1].min(), X[:, 1].max()])
    plt.plot(line_x, theta[0] + theta[1]*line_x, '--', label='fit')
    plt.plot(line_x, theta_true[0] + theta_true[1]*line_x, ':', label='signal')
    plt.legend()


# In[66]:


# Fit (learned theta) vs. Signal (true theta)
plotter_multiple(X, y, theta, theta_true)


# In[67]:


# Sub-challenge 3 cont.
# Save loss at each epoch for plotting
def GD_memory(X, y, lr, epoch):
    '''
    Finds theta from X and y using gradient descent.
    Starts by random init. of theta
    Repeats GD_one_step for predefined number of epochs
    At each epoch calculates the loss and saves it into a list
    Arguments:
        X: np array with shape (m, 1+n)
        y: np array with shape (m, 1)
        lr: learning rate, a float
        epoch: an integer
    Returns:
        theta: np array with shape (1+n, 1), final theta after training
        memory: a list of floats, each item is loss at each epoch
    '''
    memory = []
    
    return theta, memory


# In[68]:


theta, memory = GD_memory(X, y, lr=0.01, epoch=1000)


# In[69]:


def loss_plotter(memory):
    '''
    Plots the list of losses
    '''
    plt.figure(dpi=200)
    plt.plot(memory)
    plt.ylabel('loss')
    plt.xlabel('number of epochs')


# In[70]:


loss_plotter(memory)


# ## It should work with noisy data as well

# In[71]:


# let's introduce noise
y_real = y * np.random.uniform(0.97, 1.03, m).reshape(-1, 1)


# In[72]:


# Plot noisy X and y, with signal
plotter(X, y_real, theta_true)


# In[73]:


theta, memory = GD_memory(X, y_real, lr=0.01, epoch=10000)


# In[74]:


loss_plotter(memory)


# In[75]:


# Fit vs. Signal for noisy data
plotter_multiple(X, y_real, theta, theta_true)


# ## How about more features and noise

# In[76]:


# Repeat data generation with more features + noise
m = 10  # number of examples
n = 2  # number of features


# In[77]:


X = np.random.uniform(size=(m, n))
X


# In[78]:


X = np.concatenate((np.ones((m, 1)), X), axis=1)
X


# In[79]:


theta_true = np.random.uniform(size=(n+1, 1))
theta_true


# In[80]:


y = np.dot(X, theta_true)


# In[81]:


# Add noise
y = y * np.random.uniform(0.97, 1.03, m).reshape(-1, 1)


# In[82]:


theta, memory = GD_memory(X, y, lr=0.1, epoch=10000)
loss_plotter(memory)


# In[83]:


# Learned params vs. True params
print('theta: ', theta.T)
print('theta_true: ', theta_true.T)


# In[84]:


# Prediction vs. Label
print('Prediction:', np.dot(X, theta)[0])
print('Ground truth:', y[0])


# In[ ]:




