import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

# np.random.seed(1)

# GRADED FUNCTION: create_placeholders

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    
    return X, Y


# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    # 为了让每次运行时随机生成的值都是固定的，用1作为seed
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
    # he_normal(seed=None)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    # 取出参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add( tf.matmul(W1, X), b1 )
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add( tf.matmul(W2, A1), b2 )
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add( tf.matmul(W3, A2), b3 )
    
    return Z3



def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001, num_epochs = 1200, minibatch_size = 32, print_cost = True):
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    # 为 X Y 创建 placeholder
    X, Y = create_placeholders(n_x, n_y)

    # 初始化参数
    parameters = initialize_parameters()
    
    # 正向传播
    Z3 = forward_propagation(X, parameters)

    # 计算成本函数    
    cost = compute_cost(Z3, Y)
    
    # 创建优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # 初始化所有变量的操作
    init = tf.global_variables_initializer()

    # 运行计算图
    with tf.Session() as sess:
        
        # 进行初始化
        sess.run(init)
        
        # 执行训练
        for epoch in range(num_epochs):

            epoch_cost = 0.
            # 得到mini-batch的数量
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            # 随机获取一组mini-batch
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                
                # 执行优化器的操作，并返回cost。对于不想保留的返回值，我们用 _ 来占位
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # 打印cost
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # 绘制 cost 走势
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # 保存参数
        parameters = sess.run(parameters)

        # 返回一个元素值为 True 或者 False 的矩阵
        # tf.argmax(Z3) 返回Z3数据中最大值对应的下标，也就是预测的类别
        # 与Y中对应的类别加以比较，若对应位置的值相同，则返回的矩阵中该位置的值为 True
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # 将值为 True 和 False 的矩阵转换为浮点数矩阵，并计算平均值，就得到了准确度
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # 下面的语句和这一句等价：print("Train Accuracy:", sess.run(accuracy, {X: X_train, Y: Y_train}))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        # 下面的语句和这一句等价：print("Test Accuracy:", sess.run(accuracy, {X: X_test, Y: Y_test}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters




# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


parameters = model(X_train, Y_train, X_test, Y_test)