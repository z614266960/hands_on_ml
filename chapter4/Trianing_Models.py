import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# 确定随机数
np.random.seed(42)

X=2*np.random.rand(100,1)
y=4+3*X+np.random.randn(100,1)


# 最小二乘法
'''
Theta 各列系数
Theta = (X.T * X)^-1 * X.T * Y
X需要并上一个常数列，用于与theta的常数系数列相乘
'''
X_b=np.c_[np.ones((100,1)),X]  # add x0 = 1 to each instance
theta_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 测试
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new,y_predict,"r-")
plt.plot(X,y,"b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0,2,0,15])
plt.show()

# 使用sklearn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
print(lin_reg.intercept_,lin_reg.coef_)
print(lin_reg.predict(X_new))

# 批量梯度下降
'''
theta(next_step) = theta(now) - eta(learning_rate) * d(MSE(theta))/d(theta)
d(MSE(theta))/d(theta) 为 mse对 theta 的求导
d(MSE(theta))/d(theta) = 2/m X.T * (X * theta - y)
'''
eta = 0.1 #learning rate
n_iterations = 1000 #迭代次数
m=100 #个数

theta = np.random.randn(2,1) #随机给一个数进行初始化
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta*gradients
print(theta)

# 随机梯度下降
'''
1.每次选用一组x,y求gradients而不是全部数据，减小花费
2.eta学习速率先大后小
'''
n_epochs = 50
t0,t1 = 2,50

# 学习率逐渐变小
def learning_schedule(t):
    return t0/(t+t1)

theta  = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index=np.random.randint(m)
        xi=X_b[random_index:random_index+1]
        yi=y[random_index:random_index+1]
        gradients=2*xi.T.dot(xi.dot(theta)-yi)
        eta=learning_schedule(epoch*m+i)
        theta=theta-eta*gradients
print(theta)

# 使用sklearn SDG
from sklearn.linear_model import SGDRegressor

sgd_reg=SGDRegressor(n_iter=50,penalty=None,eta0=0.1)
sgd_reg.fit(X,y.ravel())
print(sgd_reg.intercept_,sgd_reg.coef_)

