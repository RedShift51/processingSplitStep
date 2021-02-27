import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import argparse
from scipy.interpolate import interp1d

""" Process slices iexp """
path_dir = "/home/abuzovkin/py_atm/big_data" #"/home/alex/TP/hse_data"
dists = [28, 56, 84, 112]#, 140, 168, 196]
storage_all = {d: {} for d in dists}
dx = 3900 / 36864.
halfs = [125., 375., 720., 1100] #[200, 400, 700, 950]
# set of distances
#half_rad = [0] + [50 * n for n in range(1, 50) if 50 * n < 1850]
#half_rad = np.array(half_rad)

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--start", default=0, type=int)
    #parser.add_argument("--n_threads", default=8, type=int)
    parser.add_argument("--begin_seed", default=1, type=int)
    parser.add_argument("--begin_dist", default=0, type=int)
    args = parser.parse_args()
    return vars(args)

def take_half(curr_dist):
    path_an = "/home/abuzovkin/py_atm/scripts/an_data"
    curr_file = [f for f in os.listdir(path_an) if f.find(str(int(curr_dist)))!=-1][0]
    print(curr_file, " curr file")
    with open(os.path.join(path_an, curr_file), "r") as f:
       data = f.readlines()[1:]
    data = np.stack([np.array([float(m) for m in s.strip().split(" ")]) \
            for s in data if len(s) > 0])

    #print(data[data[:,1] > 0.5 * data[0,1], 0])
    data = data[data[:,1] > 0.5 * data[0,1], 0][-1]
    #print(data)
    #print(data.shape)
    return data

def extract_an_vals(curr_dist, max_dist):
    path_an = "/home/abuzovkin/py_atm/scripts/an_data"
    curr_file = [f for f in os.listdir(path_an) if f.find(str(int(curr_dist)))!=-1][0]
    with open(os.path.join(path_an, curr_file), "r") as f:
       data = f.readlines()[1:]
    data = [[float(m) for m in s.strip().split(" ")] for s in data if len(s) > 0]

    if np.max([n[0] for n in data]) < max_dist:
       data.append([max_dist + 1, 1e-10])
    interCurve = interp1d([n[0] for n in data], [n[1] for n in data], kind="linear")
    #try:
    #    return interCurve(curr_r)
    #except:
    #    return -1
    return interCurve

def find_fir_zero(arr):
    for i in range(len(arr)):
        if arr[i] == 0:
            return i
    return -1

args = parse_args()
# seed
local_storage = {d: [[]] * 6 for d in dists}
numbins = 50
for d0,d in enumerate(dists):
    if d != args["begin_dist"]:
        continue

    half_curr = take_half(d)
    interCurve = extract_an_vals(d, 2000)
    half_rad = take_half(d)
    half_rad = np.array([n * half_curr / 5 for n in range(7)])

    #half_rad = [0] + [(halfs[d0] / 8.) * n for n in range(1, 50) if 50 * n < 1850]
    #half_rad = np.array(half_rad)
    for s0,s in enumerate(os.listdir(os.path.join(path_dir, "iexpN"))):
        #if s0 < 175:
        #    print(s0, s, d)
        #    continue

        print(s)
        #if int(s[1:]) != args["begin_seed"]:
        #    continue
        path_pkl_file = os.path.join("/home/abuzovkin/py_atm/big_data/iexp_calculated/", str(d),\
                            "data_" + str(d) + str(s) + ".pkl")
        if os.path.exists(path_pkl_file) is True:
            continue

        curr_file = [k for k in os.listdir(os.path.join(path_dir, "iexpN", s)) \
                         if k.find("_" + str(d))!=-1]
        if len(curr_file) == 0:
            continue
        #if os.path.exists(os.path.join("/home/abuzovkin/py_atm/big_data/iexp_calculated/", str(d),\
        #                    "data_" + str(d) + str(s) + ".pkl")) is True:
        #    continue
        print("Here")
        curr_file = curr_file[0]
        curr_file = pd.read_csv(os.path.join(path_dir, "iexpN", s, curr_file), \
                                header=None)#.transpose()
        curr_file = np.array(curr_file).T
        print(d, "km", s, "seed", curr_file.shape)
        curr_file[:, 0] *= dx
        curr_file[:, 1] *= dx

        #if os.path.exists(os.path.join("/home/abuzovkin/py_atm/big_data/iexp_calculated/", str(d),\
        #                    "data_" + str(d) + str(s) + ".pkl")) is True:
        #    continue
        
        #local_storage = []
        check_list = np.sqrt(curr_file[:,0]**2 + curr_file[:,1]**2)
        for circ in range(len(half_rad) - 1):
            print(circ)
            Iavg = interCurve((half_rad[circ] + half_rad[circ+1]) / 2)
            binsize0 = Iavg * 1e-2
            lastedge = Iavg * 1e+3
            c = np.log(1 + lastedge / binsize0) / numbins
            binedges = (np.exp(c * np.arange(numbins + 1)) - 1) * binsize0

            try:
                data = [curr_file[:,2][(check_list > half_rad[circ]) & \
                        (check_list < half_rad[circ + 1])]][:][0]
            except:
                print(d, "seed:", s, "fail")
                continue
            if len(data) == 0:
                continue

            #y, x, _ = plt.hist(data, density=True, bins=50)
            #x = (x[1:] + x[:-1]) * 0.5
            y, x = np.histogram(data, binedges, density = False)
            x2 = 0.5 * (binedges[:-1] + binedges[1:]) #np.sqrt(binedges[:-1] * binedges[1:])

            """
            pos = find_fir_zero(y)
            if pos != -1:
                y, x = y[:pos], x[:pos]
            
            lin_reg = LinearRegression()
            # find first zero
            if len(x) == 0:
                continue
            lin_reg.fit(x.reshape(-1, 1), np.log(y))
            r_dist = (half_rad[circ] + half_rad[circ + 1]) / 2.
            if np.abs(-1. / lin_reg.coef_[0]) > 1:
                continue
            """
            #local_storage[d][circ].append(-1. / lin_reg.coef_[0])
            local_storage[d][circ] = [x2, y]

            print(circ, len(half_rad) - 1)

        with open(os.path.join("/home/abuzovkin/py_atm/big_data/iexp_calculated/", str(d),\
                            "data_" + str(d) + str(s) + ".pkl"), 'wb') as f:
            pickle.dump(local_storage[d], f)

        local_storage[d] = [[]] * (len(half_rad) - 1)
