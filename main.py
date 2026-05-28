import numpy as np
import argparse
import importlib
import torch
import os

# from config import MODEL_PARAMS
# from params.adaptive.options_adapt_compas_a import arg_parser
from options import arg_parser
from optimalfair.utils.data_utils import get_data
from config import ALGORITHMS_MAPPING

def read_options():

    # read options
    args = arg_parser()

    # Set options as a dict
    try: 
        options = vars(args)
    except IOError as msg:
        args.error(str(msg))
    print(options)

    # use gpu
    if options['gpu']:
        if torch.cuda.is_available():
            options['device'] = torch.device("cuda:0")
        else:
            options['gpu'] = False
            print('gpu is unavailable')

    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])


    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    # Load selected algorithm
    algorithm_path = 'optimalfair.algorithm.%s' % ALGORITHMS_MAPPING[options['algorithm'].lower()]
    mod = importlib.import_module(algorithm_path)
    algorithm_class = getattr(mod, 'classifier')

    return options, algorithm_class

def main():
    # Parse command line arguments
    options, algorithm_class = read_options()

    # `dataset` : ( train_data, valid_data, test_data, n_group, n_class)
    data_info = get_data(options)
    selected_algorithm = algorithm_class(data_info, options)
    selected_algorithm.train()

if __name__ == '__main__':
    main()