import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import os
from options import args_parser

from matplotlib import rcParams
rcParams.update({'font.size': 18, 'text.usetex': True})

file_path = os.path.dirname(__file__)


def plot_logreg(args):
    '''
    logistic regression: weights
    '''

    logreg_optimizer_weights, logreg_optimizer_objective = pkl.load(
        open(file_path + '/results/logreg_' + args.optimizer + '.pkl', 'rb'))

    logreg_dimension = 785
    plt.figure()
    plt.plot(range(len(logreg_optimizer_weights)), np.array(
        logreg_optimizer_weights) / np.sqrt(logreg_dimension), label=args.optimizer)

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$\frac{1}{\sqrt{d}}\|x^{(k)}-x^{\star}\|_2$')
    plt.title('Logistic Regression')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(file_path + '/results/logreg_' + args.optimizer + '_weights.png', dpi=1200)
    plt.show()
    plt.pause(5)


    '''
    logistic regression: objective
    '''

    plt.figure()
    plt.plot(range(len(logreg_optimizer_objective)), logreg_optimizer_objective, label=args.optimizer)

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$f(x^{(k)}) - p^{\star}$')
    plt.title('Logistic Regression')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(file_path + '/results/logreg_' + args.optimizer + '_objective.png', dpi=1200)
    plt.show()


def plot_all_optimizer():

    optimizers = ['GD', 'MN', 'CG', 'LM', 'BFGS']

    figure, axis = plt.subplots(1, 2, figsize=(16, 6))

    logreg_dimension = 785
    for optimizer in optimizers:

        # logistic regression: weights and objective
        logreg_optimizer_weights, logreg_optimizer_objective = pkl.load(
            open(file_path + '/results/logreg_' + optimizer + '.pkl', 'rb'))

        axis[0].plot(range(len(logreg_optimizer_weights)), np.array(
            logreg_optimizer_weights) / np.sqrt(logreg_dimension), label=optimizer)
        axis[0].grid(True)

        axis[1].plot(range(len(logreg_optimizer_objective)), logreg_optimizer_objective, label=optimizer)
        axis[1].grid(True)

    axis[0].set(xlabel='Iterations', ylabel=r'$\frac{1}{\sqrt{d}}\|x^{(k)}-x^{\star}\|_2$')
    axis[0].set_title('Logistic Regression Weights')
    axis[0].legend()
    axis[0].set_yscale('log')

    axis[1].set(xlabel='Iterations', ylabel=r'$f(x^{(k)}) - p^{\star}$')
    axis[1].set_title('Logistic Regression Objective')
    axis[1].legend()
    axis[1].set_yscale('log')

    figure.savefig(file_path + '/results/logreg_all_weights_and_objective.png', dpi=1200)


if __name__ == '__main__':
    # Running from the command line
    args = args_parser()

    plot_logreg(args)
