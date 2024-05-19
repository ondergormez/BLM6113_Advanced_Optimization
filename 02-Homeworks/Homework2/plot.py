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

if __name__ == '__main__':
    # Running from the command line
    args = args_parser()

    plot_logreg(args)
