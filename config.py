# Global parameters
DATASETS = ['mnist', 'synthetic', 'emnist', 'fmnist', 'cifar']

MODELS = ['logistic', '2nn', '1nn', 'resnet']

ALGORITHMS_MAPPING = {
                      'linearpost':'LinearPost',
                      'erm':'ERM',
                      'werm':'weightERM',
                      'infair': 'infair',
                      'postfair': 'postfair',
                      'postfair_a':'postfair_a',
                      'fairproj': 'FairProjection',
                      'fdiv': 'F-divergence',
                      'advbias': 'Advdebias',
                      'inpost': 'inpostfair',
                     }

OPTIMIZERS = ALGORITHMS_MAPPING.keys()
