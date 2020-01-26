import pandas as pd
import numpy as np
import os

def simulate(input_data, effect, out=None):
    """
    simulate dataset with known effect to test implemented methods
    :param input_data: a dataframe
    :param effect: the effect of the treatment
    :param output: output file path.
    :return: simulated data frame containing t,y0,y1 and y
    """

    # data standardization
    input_data = (input_data - np.mean(input_data, axis=0))/np.std(input_data, axis=0)

    # create a bias matrix to tie between treatment and random data generation
    strong_bound = np.random.uniform(0.7,0.95)
    weak_bound = np.random.uniform(0.1, 0.35)
    bias = np.array(((strong_bound, weak_bound),(weak_bound, strong_bound)))

    # generate treatment and Y0 vector
    T = np.matmul(np.random.normal(size=(input_data.shape[1],2)), bias)
    TY = np.matmul(input_data.values, T)

    input_data["T"] = (TY[:,0] > 0).astype("int")
    input_data["Y0"] = TY[:,1]
    input_data["Y1"] = input_data["Y0"] + effect
    input_data["Y"] = input_data["Y0"]*(1 - input_data["T"]) + input_data["Y1"]*input_data["T"]

    if out is not None:
        input_data.to_csv(os.path.join(out, 'simulated_input.csv'), index=False)

    return input_data
