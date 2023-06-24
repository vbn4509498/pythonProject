import numpy as np

np.set_printoptions(threshold=np.inf)
def compute_error_for_line_given_points(b,w,points):   ##舉例 points = [100,2] 100為有100個數據 2為 x 和 y 兩變數
    totalError = 0
    print(b,w)
    for i in range(1, len(points)):

        x = points[i, 0] # 0 為x
        y = points[i, 1] # 1 為y

        totalError += (y-(w*x+b))**2

    return totalError / float(len(points))

def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]  # 0 為x
        y = points[i, 1]  # 1 為y
        # grad_b = 2(wx+b-y)

        b_gradient += (2/N)*((w_current * x + b_current)-y)
        # grad_w = 2(wx+b-y)*x # /N 為平均值
        w_gradient += (2/N) * x * ((w_current * x + b_current)-y)
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    #if np.isinf(new_b):
        #new_b = b_current
    #elif np.isinf(new_w):
        #new_w = w_current
    return [new_b,new_w]

def gradient_descent_runner(points, starting_b ,starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
        print(b,w)
    return[b, w]

def run():
    points = np.genfromtxt('data.csv',delimiter=",",encoding="utf-8")
    points[0,0]  =87.28955302
    learning_rate = 0.001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0} , w = {1} , error = {2}".format(initial_b,initial_w,
    compute_error_for_line_given_points(initial_b,initial_w,points)))
    print("Running...")
    [b,w] = gradient_descent_runner(points,initial_b,initial_w,learning_rate,num_iterations)
    print([b,w])
    print("After {0} iterations b = {1} , w = {2} , error = {3}".format(num_iterations,b,w,compute_error_for_line_given_points(b,w,points)))
if __name__ == "__main__":
    run()