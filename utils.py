import os
import pandas as pd
#import matplotlib.pyplot as plt

import traceback
import contextlib

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))


def plot_learning_curve(train_loss, val_loss, metric_name='mse'):
    plt.plot(train_loss, color='r')
    plt.plot(val_loss, color='b')
    plt.legend(['training', 'validation'])
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)

def plot_learning_curve_from_history(training_history, validation_history, output_folder, metric_name='mse'):
    train_metric_name = 'train_'+ metric_name
    val_metric_name = 'val_' + metric_name
    for drug in training_history[train_metric_name]:
        plt.figure()
        plt.plot(training_history[train_metric_name][drug], color='r')
        plt.plot(validation_history[val_metric_name][drug], color='b')
        plt.legend(['training', 'validation'])
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(drug)
        plt.savefig(os.path.join(output_folder, drug+'_'+metric_name+'.png'))


def best_per_drug(best_epoch_dict, validation_history, metric_name='pearson'):
    metric_key_name = 'val_' + metric_name
    best_metric_per_drug_df = pd.DataFrame.from_dict(best_epoch_dict, columns=[metric_name])
    for drug in best_epoch_dict:
        best_metric_per_drug_df.loc[drug, metric_name] = validation_history[metric_key_name][drug][best_epoch_dict[drug]]

    return best_metric_per_drug_df


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')

def list_to_repr(nums):
    result = ''
    for num in nums:
        result += repr(num) + '_'

    return result