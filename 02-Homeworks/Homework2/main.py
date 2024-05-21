import os
import numpy as np
import pickle as pkl
from datasets.data_preprocess import data_preprocess
import logistic_regression
from options import args_parser
import yaml
import logging.config
import logging
import sys

sys.path.append('./datasets')

current_work_dir = os.path.dirname(__file__)

with open(current_work_dir + '/logger_config.yaml', 'r') as file:
    config = yaml.safe_load(file.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)


def main(args):

    (x_train, y_train), (x_test, y_test) = data_preprocess(args)

    logger.info('learning rate: {0}'.format(args.lr))
    logger.info('Optimizer: {0}'.format(args.optimizer))

    Model = logistic_regression.LogisticRegression(args=args, X_train=x_train, Y_train=y_train,X_test=x_test)

    weight_diff_list = []
    obj_diff_list = []
    weight_diff, obj_diff = Model.diff_cal(Model.weights)
    logger.info("------------ Initial ------------")
    logger.info("weight error: {:.4e}".format(weight_diff))
    logger.info("objective error: {:.4e}".format(obj_diff))

    Eigvals = np.linalg.eigvals(Model.pre_Hessian)
    logger.info("max eigenvalue of Hessian:{:.4f}".format(np.max(Eigvals)))
    logger.info("min eigenvalue of Hessian:{:.4f}".format(np.min(Eigvals)))

    '''
    update
    '''
    for i in range(args.iteration):

        weight_diff, obj_diff = Model.update()

        if i % 50 == 0 or i == args.iteration - 1:
            logger.info("------------ Iteration {} ------------".format(i+1))
            logger.info("weight error: {:.4e}".format(weight_diff))
            logger.info("objective error: {:.4e}".format(obj_diff))
        else:
            logger.debug("------------ Iteration {} ------------".format(i+1))
            logger.debug("weight error: {:.4e}".format(weight_diff))
            logger.debug("objective error: {:.4e}".format(obj_diff))

        weight_diff_list.append(weight_diff)
        obj_diff_list.append(obj_diff)

        if weight_diff / np.sqrt(Model.dimension) <= 1e-5:
            # print last iteration result
            logger.info("------------ Iteration {} ------------".format(i+1))
            logger.info("weight error: {:.4e}".format(weight_diff))
            logger.info("objective error: {:.4e}".format(obj_diff))
            break

    if args.is_armijo_stepsize_active:
        file_name = './results/{}_{}_armijo.pkl'.format('logreg', args.optimizer)
    else:
        file_name = './results/{}_{}.pkl'.format('logreg', args.optimizer)
    file_name = os.path.join(current_work_dir, file_name)

    val = Model.getTest() > 0.5
    val2 = y_test > 0.5
    correct = 0

    for i in range(len(val2)):
        if val[i] == val2[i]:
            correct += 1

    percent_correct = correct / len(val2) * 100
    logger.warning('Accuracy: {0:.3f}%'.format(percent_correct))


    with open(file_name, 'wb') as f:
        pkl.dump([weight_diff_list, obj_diff_list], f)


if __name__ == '__main__':
    # Running from the command line
    args = args_parser()

    main(args)
