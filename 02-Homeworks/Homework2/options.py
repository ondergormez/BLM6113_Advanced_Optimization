import argparse
from argparse import RawTextHelpFormatter

def args_parser():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--is_armijo_stepsize_active', type=bool, choices=[True, False], default=False,
                        help="Armijo stepsize activation (default: False)")
    parser.add_argument('--lr', type=float, default = 0.1,
                        help="learning rate for each update step (default: 0.1)")
    parser.add_argument('--optimizer', type=str, choices=['GD', 'MN', 'CG', 'LM', 'BFGS'], default='GD',
                        help="Choose an optimizer;\n"
                        "GD: Gradient Descent (default)\n"
                        "MN: Modified Newton\n"
                        "CG: Conjugate Gradient\n"
                        "LM: Levenberg-Marquardt\n"
                        "BFGS: Broyden-Fletcher-Goldfarb-Shanno")
    parser.add_argument('--iteration', type=int, default = 250,
                        help="maximum update iterations if not exit automatically (default: 250)")
    parser.add_argument('--gamma', type=float, default=0.1,
                        help="penalty term for logistic regression (default: 0.1)")

    parser.add_argument('--f', type=str, default='',
                        help="Dummy argument for running the program from the Jupyter notebook. " +
                             "Don't change it if you are using the command line.")

    args = parser.parse_args()
    return args
