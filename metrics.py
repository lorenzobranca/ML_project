import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def score(y_pred,y_true):

    R_2=r2_score(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    mae=mean_absolute_error(y_true,y_pred)
    print('R2:',R_2)
    print('mse:',mse)
    print('mae',mae)

    return R_2, mse, mae
