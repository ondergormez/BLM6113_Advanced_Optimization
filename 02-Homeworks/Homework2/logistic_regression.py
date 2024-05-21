from numpy.linalg import inv
import numpy as np
import cvxpy as cp
epsilon = 1e-5


class LogisticRegression():

    def __init__(self, args, X_train, Y_train,X_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        
        self.num_samples = self.X_train.shape[0]
        self.dimension = self.X_train.shape[1]

        self.weights = np.zeros_like(X_train[0])
        self.is_armijo_stepsize_active = args.is_armijo_stepsize_active
        self.lr = args.lr
        self.optimizer = args.optimizer

        self.gamma = args.gamma

        print("============= CVX solving =============")
        self.opt_weights, self.opt_obj = self.CVXsolve()
        print("============= CVX solved =============")

    
        self.pre_Hessian = self.Hessian(self.weights)
    
    def sigmoid(self, input):
        return 1./(1.+ np.exp(-input))

    def getTest(self):
        self.test = self.sigmoid(self.X_test @ self.weights)
        return self.test

    def objective(self, weights):
        '''
        return the objective value of the problem
        note that the objective is averaged over all samples
        '''
        sigWeights = self.sigmoid(self.X_train @ weights)
        matGrad = self.Y_train * np.log(sigWeights+epsilon) + (1.0 - self.Y_train) * np.log(1-sigWeights+epsilon)
        return  - np.sum(matGrad) / self.num_samples + 0.5*self.gamma*np.linalg.norm(weights)**2

    def gradient(self, weights):
        '''
        return the gradient of objective function
        note that the gradient is averaged over all samples
        '''
        sigWeights = self.sigmoid(self.X_train @ weights)
        matGrad = self.X_train.T @ (sigWeights - self.Y_train)
        return matGrad / self.num_samples + self.gamma*weights

    def Hessian(self, weights):
        '''
        return the Hessian of objective function
        note that the Hessian is averaged over all samples
        '''
        sigWeights = self.sigmoid(self.X_train @ weights)
        D_diag = np.diag(sigWeights * (1-sigWeights))
        return self.X_train.T @ D_diag @ self.X_train / self.num_samples + self.gamma * np.identity(self.dimension)


    def update(self):

        '''
        update model weights using GD  step
        '''
        gradient = self.gradient(self.weights)

        if self.optimizer == 'GD':
            if self.is_armijo_stepsize_active:
                self.lr = self.armijoStepLengthController(gradient)

            update_direction = gradient
            self.weights -= self.lr * update_direction
        elif self.optimizer == 'MN':
            if self.is_armijo_stepsize_active:
                self.lr = self.armijoStepLengthController(gradient)

            hessian = self.Hessian(self.weights)
            inverse_hessian = inv(hessian)

            S = -1 * np.dot(inverse_hessian, gradient)
            self.weights = self.weights + self.lr * S
        elif self.optimizer == 'CG':
            if self.is_armijo_stepsize_active:
                self.lr = self.armijoStepLengthController(gradient)

            update_direction = -1 * gradient
            updated_weights = self.weights + self.lr * update_direction

            gradient_2 = self.gradient(updated_weights)
            S2 = -1 * gradient_2 + (np.linalg.norm(gradient_2, ord=2) /
                                    np.linalg.norm(gradient, ord=2)) * update_direction
            self.weights = updated_weights + self.lr * S2
        elif self.optimizer == 'LM':
            hessian = self.Hessian(self.weights)
            I = np.identity(self.weights.shape[0])

            S = -1 * np.dot(inv(hessian + self.lr * I), gradient)
            self.weights = self.weights + self.lr * S
        elif self.optimizer == 'BFGS':
            hessian = self.Hessian(self.weights)
            inverse_hessian = inv(hessian)

            S1 = -1 * np.dot(inverse_hessian, gradient)
            w2 = self.weights + self.lr * S1

            g2 = self.gradient(w2)
            delta_w = w2 - self.weights
            delta_g = g2 - gradient

            I = np.identity(self.weights.shape[0])
            delta_w = delta_w.reshape(len(delta_w), 1)
            delta_g = delta_g.reshape(len(delta_w), 1)

            R = 1 / delta_w.T.dot(delta_g)
            num = inverse_hessian.dot(delta_g)
            v1 = 1 + R * num.T.dot(delta_g)
            v2 = R * np.outer(delta_w, delta_w)
            v3 = R * np.outer(delta_w, num)
            v4 = R * np.outer(num, delta_w)
            G2 = v1 * v2 - v3 - v4

            self.weights = w2
        else:
            raise NotImplementedError

        a, b = self.diff_cal(self.weights)
        return a,b

    # Modified from: https://github.com/saurabbhsp/machineLearning/blob/master/LinearRegression-GradientDescent/GradientDescent.py#L127x
    def armijoStepLengthController(self, gradient):

        np.random.seed(8)
        beta = np.random.random_sample(785)
        delta = 0.2

        x = self.X_train
        x = x * 1.0

        y = self.Y_train
        y = y * 1.0

        y_prediction = np.dot(beta, x.T)
        residual = y_prediction - y
        fx = np.dot(residual.T, residual)

        maxIterations = 50
        alpha = 1.0
        gradientSquare = np.dot(gradient, gradient)

        for i in range(0, maxIterations):

            alpha = alpha/2

            residual_alpha_gradient = y - np.dot((beta - (alpha * gradient)), x .T)
            fx_alpha_gradient = np.dot(residual_alpha_gradient.T, residual_alpha_gradient)

            """Convergence condition for armijo principle"""
            if fx_alpha_gradient < fx - (alpha * delta * gradientSquare):
                break

        return alpha


    def CVXsolve(self):
        '''
        use CVXPY to solve optimal solution
        '''
        x = cp.Variable(self.dimension)
        objective = cp.sum(cp.multiply(self.Y_train, self.X_train @ x) - cp.logistic(self.X_train @ x))
        prob = cp.Problem(cp.Maximize(objective/self.num_samples - 0.5*self.gamma*cp.norm2(x)**2))
        prob.solve(solver=cp.ECOS_BB, verbose=False) # False if not print it

        opt_weights = np.array(x.value)
        opt_obj = self.objective(opt_weights)

        return opt_weights, opt_obj

    def diff_cal(self, weights):
        '''
        calculate the difference of input model weights with optimal in terms of:
        -   weights
        -   objective
        '''
        weight_diff = np.linalg.norm(weights - self.opt_weights)
        obj_diff = abs(self.objective(weights) - self.opt_obj) 
        return weight_diff, obj_diff
