import os
import logging
import argparse
import configparser


parameters = [
    # Global
    ("log_header", (str, "Trial New Header Again", "Header to put at the top of start of log of script")),

    # Model Parameters
    ("conv_filters", (int, 64, "Number of convolution filters in the first convoluton layer")),
    ("depth", (int, 6, "Depth of the network")),
    ("batchnorm", (int, 1, "Batch norm to be used in model. Give binary input")),

    # Hyper Parameters
    ("epochs", (int, 300, "Number of epochs to train the model on")),
    ("batchsize", (int, 16, "Batch size for training")),
    ("shuffle", (int, 1, "Input data to be shuffled or not. Give binary input 1 or 0")),
    ("seed", (int, 1, "Seed for the random shuffle of the train-validation split")),
    ("validation_frac", (float, 0.2, "Fraction of training data to be used for validation")),
    ("optimizer", (str, 'adam', "Optimizer: sgd, rmsprop, adam")),
    ("lrstart", (float, 1e-3, "Initial learning rate")),
    ("lrscheduler", (str, 'steplr', "Learning Rate Scheduler: steplr or exponentiallr")),
    ("momentum", (float, 0.95, "Momentum for the SGD optimizaer")),
    ("decay", (float, 100.0, "Learning rate decay for exponential lr, if steplr is used, then this will denote the steps after which we need to reduce the lr")),
    ("weight_decay", (float, 0, "weight_decay for optimizers")),
    ("lambda_loss", (float, 0.5, "Lambda value for the loss function")),
    ("loss_type", (str, 'dice_combo', "Type of the loss function")),
    ("mode", (str, 'transpose', "Mode to be used during Upsampling")),
    ("regularization", (str, 'none', "Type of regularization to be applied")),
    ("reg_lamda1", (float, 0.001, "Lambda value for L1 regularization")),
    ("reg_lamda2", (float, 0.001, "Lambda value for L2 regularization")),

    # Files
    ("datadir", (str, "RVData/TrainingSet", "Directory containing the patient data (patientxx/ directory)")),
    ("model_save_dir", (str, ".", "Directory to save all the models after a fixed number of epochs")),
    ("save_epochs", (int, 100, "Save model after every saveEpochs epochs")),
    ("logfile", (str, "Name of your log file", "Name of your log file"))
]


noninitialized = {
    'learning_rate': 'getfloat',
    'momentum': 'getfloat',
    'decay': 'getfloat',
    'seed': 'getint',
}


def update_from_configfile(args, default, config, section, key):
    # Point of this function is to update the args Namespace.
    value = config.get(section, key)
    if value == '' or value is None:
        return

    # Command-line arguments override config file values
    if getattr(args, key) != default:
        return

    # Config files always store values as strings -- get correct type
    if isinstance(default, bool):
        value = config.getboolean(section, key)
    elif isinstance(default, int):
        value = config.getint(section, key)
    elif isinstance(default, float):
        value = config.getfloat(section, key)
    elif isinstance(default, str):
        value = config.get(section, key)
    elif isinstance(default, list):
        # special case (HACK): loss-weights is list of floats
        string = config.get(section, key)
        value = [float(x) for x in string.split()]
    elif default is None:
        # values which aren't initialized
        getter = getattr(config, noninitialized[key])
        value = getter(section, key)
    setattr(args, key, value)


def parse_arguments():
    parser = argparse.ArgumentParser()

    for argument, params in parameters:
        d = params
        if isinstance(params, tuple):
            d = dict(zip(['type', 'default', 'help'], params))
        parser.add_argument("--"+argument, **d)

    parser.add_argument('configfile', nargs='?', type=str, help="Load options from config")
    args = parser.parse_args()

    if args.configfile:
        # logging.critical("\n\nLoading options from config file: {}".format(args.configfile))
        config = configparser.ConfigParser(inline_comment_prefixes=['#', ';'], allow_no_value=True)
        config.read(args.configfile)
        for section in config:
            for key in config[section]:
                if key not in args:
                    raise Exception("Unknown option {} in config file.".format(key))
                update_from_configfile(args, parser.get_default(key),
                                       config, section, key)

    return args

parse_arguments()
