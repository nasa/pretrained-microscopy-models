import matplotlib.pyplot as plt

import numpy as np
import torch
from scipy.stats import linregress

def visualize_curve(train_data, val_data, start_range=0, save_fig=False, **kwargs):
    
    num_epochs = len(train_data) + 1
    x = range(1, num_epochs)
    
    plt.plot(x[start_range:], train_data[start_range:], label='Train Loss')
    plt.plot(x[start_range:], val_data[start_range:], label='Validation Loss')
    plt.legend()
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if save_fig == True:
        plt.savefig(f'{kwargs["output_path"]}')
    else:
        plt.show()
        
    plt.clf()
    
def prediction_scatter(train_loader, val_loader, model, save_fig=False, **kwargs):
    # create storage lists
    train_prediction_list = []
    train_target_list = []
    test_prediction_list = []
    test_target_list = []
    
    # prediction sets
    loaders = [train_loader, val_loader]

    # make predictions
    with torch.no_grad():
        for loader in loaders:
            for data in loader:
                # send input and targets to gpu for prediction
                images, targets = data
                images, targets = images.cuda(), targets.cuda()
                # make prediction
                outputs = model(images)

                # bring predictions back to cpu
                outputs_arr = outputs.cpu().numpy()
                targets_arr = targets.cpu().numpy()

                if loader == loaders[0]:
                    train_prediction_list += list(outputs_arr[:,0])
                    train_target_list += list(targets_arr[:,0])
                else:
                    test_prediction_list += list(outputs_arr[:,0])
                    test_target_list += list(targets_arr[:,0])

    train_pred_arr = np.array(train_prediction_list)
    train_target_arr = np.array(train_target_list)
    test_pred_arr = np.array(test_prediction_list)
    test_target_arr = np.array(test_target_list)
    
    plt.scatter(test_pred_arr, test_target_arr, alpha=0.5, color='yellow', label='Validation')
    plt.scatter(train_pred_arr, train_target_arr, alpha=0.1, color='blue', label='Train')
    plt.legend()
    
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    plt.title('True vs Predicted Value')
    
    if save_fig == True:
        plt.savefig(f'{kwargs["output_path"]}')
    else:
        plt.show()
    
    plt.clf()
    
def r2_plot(df_summary, feature, statistic, save_fig=False, **kwargs):
    slope, intercept, r_value, p_value, std_err = linregress(df_summary[feature], df_summary[statistic])
    r_squared = r_value ** 2
    
    fit_line = slope * df_summary[feature] + intercept
    
    plt.scatter(df_summary[feature], df_summary[statistic])
    plt.plot(df_summary[feature], fit_line, color='red')
    
    plt.xlabel('Normalized Metric')
    plt.ylabel(f'{statistic} Metric')
    
    if save_fig == True:
        plt.savefig(f'{kwargs["output_path"]}')
    else:
        plt.show()
        
    plt.clf()
    
    return r_squared