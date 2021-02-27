import os
import numpy as np
import pickle as pkl
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d

#dists = [28, 56, 84, 112, 140, 168, 196]
#means = {str(k): {"data": np.zeros((36864, 36864), dtype=np.float64), "count": 0} \
#            for k in dists}
halfs = [6.75, 18.0, 45.0, 82.5, 127.5] #[127, 200, 350, 500]
N =  36864 #16384 #8192#36864
dx = 3900. / 36864 #1733.3333 / N #819.2 / N #3900. / N
"""
rads = np.array([0, 25, 35, 79, 80, 138, 135, 179, 197, 213, 228, 243, 270, \
                300, 330, 360, 390, 420, 450, 480, 510, 540, \
                570, 600, 630, 660, 700, 750, 800, 850, 900, 950, \
                1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400, \
                1500, 1600, 1700, 1800]) / dx
rads = rads[::2]
rads = rads[:19]
"""
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


#func_inter = extract_an_vals(ds / 1000., np.max(r))
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int)
    #parser.add_argument("--n_threads", default=8, type=int)
    #parser.add_argument("--begin_seed", default=1, type=int)
    #parser.add_argument("--begin_dist", default=0, type=int)
    args = parser.parse_args()
    return vars(args)

def iteration(arr, mask, i_brd, o_brd, Iavg):
    mask_local = np.where((mask < o_brd) & (mask >= i_brd))
    ans = [arr[mask_local].ravel()][:][0]
    #return plt.hist(ans, bins=50, density=True)#np.mean(ans), np.std(ans)

    #fig, ax = plt.subplots()
    #data = ax.hist(ans[ans > 0], bins=250)
    #plt.close()

    #numbins = 250
    #binsize = 0.01 * Iavg
    #binedges = np.arange(numbins) * binsize
    #data = np.histogram(ans, binedges)
    numbins = 50
    binsize0 = Iavg * 1e-2
    lastedge = Iavg * 1e+3
    c = np.log(1 + lastedge / binsize0) / numbins
    binedges = (np.exp(c * np.arange(numbins + 1)) - 1) * binsize0
    binedges[-1] = Iavg * 1e+8

    y, x = np.histogram(ans, binedges, density = False)
    x2 = np.sqrt(binedges[:-1] * binedges[1:])

    return [y, binedges] #data

def slices_stats(screen, dd, dist_curr, interp_func, rads):
    #numbins = 50
    #Iavg = np.average(screen)
    #binsize0 = 0.01 * Iavg
    #lastedge = 1000. * Iavg
    #c = np.log(1. + lastedge / binsize0) / numbins
    #binedges = (np.exp(c * np.arange(numbins + 1)) - 1) * binsize0
    #(P2, x) = np.histogram(Iring, binedges)
    #x2 = np.sqrt( binedges[:-1] * binedges[1:] )

    # Hardcode
    N = 36864 #16384 #8192 #36864
    dx = 3900. / 36864 #1733.3333 / N #819.2 / N
    #rads = np.array([halfs[dd] / 40. * n for n in range(int(450 / (halfs[dd] / 40.)))]) / dx
    #rads = np.array([0, 25, 45, 79, 112, 138, 160, 179, 197, 213, 228, 243, 270, \
    #            300, 330, 360, 390, 420, 450, 480, 510, 540, \
    #            570, 600, 630, 660, 700, 750, 800, 850, 900, 950, \
    #            1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400, \
    #            1500, 1600, 1700, 1800]) / dx
    widths = np.array([5., 1.076, 0.75, 0.537, 0.323]) * 3

    mask = np.array([k - N / 2 for k in np.arange(N)])
    mask = np.tile(mask, (N, 1))
    mask = np.sqrt(mask * mask + mask.T * mask.T)
   
    print("Starting parallel ", mask.shape, " | tasks ", len(rads) - 1) 
    result = Parallel(n_jobs=8, backend="threading", verbose=10)(delayed(iteration)\
                (screen, mask, rads[n], rads[n + 1], \
                    interp_func((rads[n]+rads[n + 1]) * dx / 2)) for n in range(len(rads) - 1))
    #return [(np.array(rads[1:]) + np.array(rads[:-1])) / 2., \
    #        np.array(list(result))]
    means, stds = [k[0] for k in list(result)], [k[1] for k in list(result)]
    return [(np.array(rads[1:]) + np.array(rads[:-1])) * dx / 2., \
            means, stds]


def main():
    #dists = [3.5, 7.0, 14.0, 21.0, 28.0]#[28, 56, 84, 112]#, 140, 168, 196]
    dists = [28, 56, 84, 112]
    #if not os.path.exists("../big_data/big_crossections"):
    #    os.mkdir("../big_data/big_crossections")
    #if not os.path.exists("big_data/stats"):
    #    os.mkdir("../big_data/stats")

    """ Firstly we have to collect statistics """
    path_dir = "/home/abuzovkin/atmosphere/screens/36/scr"
    #path_dir = "/home/abuzovkin/atmosphere/screens/8192_eq"
    #"/home/abuzovkin/atmosphere/screens/36/scr"
    l = len(os.listdir(path_dir))
    intro = parse_args()

    """ Firstly calculate averages """
    big_crossects = {str(k): {fold: np.zeros((N,), dtype=np.float64) \
                for fold in os.listdir(path_dir)} for k in dists}
    for d0,d in enumerate(dists):
        func_inter = extract_an_vals(d, N * dx / 2 + 1)
        #func_inter = extract_an_vals(d, 8192. * dx / 2)
        half_curr = take_half(d)
        rads = np.array([n * half_curr / 10 for n in range(14)]) / dx
        rads = rads[rads < N / 2.]

        #if d0 < intro["start"]:
        #    continue
        #if os.path.exists() is True:
        #    continue

        avg_arr = np.zeros((N, N), dtype=np.float64)
        count = 0
        for i0, fold in enumerate([m for m in os.listdir(path_dir)]):
            #if int(fold) > 35:
            #    continue
            if os.path.exists(os.path.join("../big_data/stats/36864_eq", \
                    str(d)) + "_f" + str(fold)) is True:
                continue

            print(d, fold)
            #if int(fold) < 301:
            #    continue;
            #print(os.listdir('.'))
            #if not os.path.exists("../big_data/big_crossections/_" + fold):
            #    os.mkdir("../big_data/big_crossections/_" + fold)
            #if not os.path.exists("../big_data/stats/_" + fold):
            #    os.mkdir("../big_data/stats/_" + fold)

            try:
                curr_file = [k for k in os.listdir(os.path.join(path_dir, fold)) \
                            if k.find(str(int(d * 1000))) != -1][0]
                temp_arr = np.reshape(np.fromfile(os.path.join(path_dir, fold, curr_file), \
                                dtype=np.float64), [N, N])
                temp_arr = temp_arr.astype(np.float32)
                avg_arr += temp_arr
                count += 1
                #np.save(os.path.join("../big_data/big_crossections", fold, str(d)), \
                #            temp_arr[int(36864 / 2), :])
            except:
                print("Fold " + fold + " error")
                continue

            direct_res = slices_stats(temp_arr, d0, d, func_inter, rads)
            #with open(os.path.join("../big_data/stats", "_" + fold, str(d)), "wb") \
            #        as ftemp:
            #    pkl.dump(direct_res, ftemp)
                #ftemp.write(direct_res)
            
        #avg_arr /= [np.array([count]).astype(np.float64)][:][0][0]
        #direct_res = slices_stats(avg_arr)
            with open(os.path.join("../big_data/stats/36864_eq", str(d)) + "_f" + str(fold), "wb") \
                    as ftemp:
                pkl.dump(direct_res, ftemp)
            #ftemp.write(direct_res)
            print(fold, d)


if __name__ == "__main__":
    main()

