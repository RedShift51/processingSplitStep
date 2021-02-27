import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
import datetime
from giaflucs import *
import time
import argparse

# we look randomly to avoid bad assumptions
def process_cluster(xs, ys, vals, dr):
    xs = np.array(xs)
    ys = np.array(ys)
    vals = np.array(vals)

    idxs = np.zeros((len(vals),))
    idxs[np.argmax(vals)] = 1

    print("tree filling begin", datetime.datetime.now())
    # it is not efficient method, it's reasonable to rewrite it
    tree = KDTree(np.concatenate([np.expand_dims(xs, -1), np.expand_dims(ys, -1)], \
            axis=-1), leaf_size=1)
    print("tree searching begin", datetime.datetime.now())
    addition = tree.query_radius(np.concatenate([np.expand_dims(xs, -1), np.expand_dims(ys, -1)], \
            axis=-1), dr, count_only=False, return_distance=False)
    print("argsorting begin", datetime.datetime.now())
    order = np.argsort(vals)
    order = len(order) - 1 - order

    print("cluster searching begin", datetime.datetime.now())
    for i in range(len(addition)):
        if len([k for k in [addition[i]][:][0] if order[k] > order[i]]) != 0:
            continue
        else:
            idxs[i] = 1
    print("cluster processing completed", datetime.datetime.now())

    return [[xs[idxs==1]][:][0], [ys[idxs==1]][:][0], [vals[idxs==1]][:][0]]


def process(curr_data, labels, cluster, dist):
    if cluster == -1:
        return [np.array(k) for k in [curr_data[labels==cluster,:]][:][0]]
    else:
        all_pos = []
        aux_array = [curr_data[labels==cluster]][:][0]
        aux_array = process_cluster(aux_array[:, 0], aux_array[:, 1], \
                            aux_array[:, 2], dist)
        for count in range(len(aux_array[0])):
            all_pos.append(np.array([aux_array[0][count], \
                                    aux_array[1][count], aux_array[2][count]]))
        return np.stack(all_pos)


def filtered_elems(x0, y0, dist, chosen_data, threshold):
    all_pos = []
    global_max = [np.max(chosen_data)][:][0]
    conv_zip = [np.stack(list(zip(x0, y0)))][:][0]
    #tree = KDTree([conv_zip][:][0], leaf_size=1)
    chosen_data = [np.stack(list(zip(x0, y0, chosen_data)))][:][0]
    print("Chosen data ", chosen_data.shape)

    sch = 0
    if threshold == -1:
        max_el = np.median(chosen_data[:, -1])
        max_el1 = np.median(chosen_data[chosen_data[:,-1] >= max_el, -1])
        max_el2 = np.median(chosen_data[chosen_data[:,-1] >= max_el1, -1])
        max_el3 = np.median(chosen_data[chosen_data[:,-1] >= max_el2, -1])
        threshold = max_el3
    else:
        threshold *= 2
    """
    try:
        max_el = np.median(chosen_data[:, -1]) #[np.max(chosen_data[:, -1])][:][0]
        max_el1 = np.median(chosen_data[chosen_data[:,-1] >= max_el, -1])
        max_el2 = np.median(chosen_data[chosen_data[:,-1] >= max_el1, -1])
        max_el3 = np.median(chosen_data[chosen_data[:,-1] >= max_el2, -1])
    except:
        print("Length of all_pos is " + str(len(all_pos)))
        #break
    """
    curr_bool = chosen_data[:,-1] >= threshold #max_el3
    curr_data = [chosen_data[curr_bool, :]][:][0]
    #[np.stack(list(zip(x0[curr_bool], y0[curr_bool], \
    #            chosen_data[curr_bool])))][:][0]
    #print(curr_data.shape)
    curr_data = curr_data[np.argsort(curr_data[:,-1]), :]
    #print(curr_data.shape)
    if curr_data.shape[0] == 0:
        print("Shape is zero")
        #break

    clustering = DBSCAN(eps=dist, min_samples=1, n_jobs=8).fit_predict(\
                        [curr_data[:, : -1]][:][0])
    uni_labels = np.unique([clustering][:][0])
    labels = np.array([clustering][:][0])

    """
    for cluster in uni_labels:
        if cluster == -1:
            #all_pos.extend([np.array(k) for k in [curr_data[labels==cluster, :]][:][0]])
            all_pos += [np.array(k) for k in [curr_data[labels==cluster, :]][:][0]]
        else:
            aux_array = [curr_data[labels==cluster]][:][0]
            aux_array = process_cluster(aux_array[:, 0], aux_array[:, 1], \
                            aux_array[:, 2], dist)
            for count in range(len(aux_array[0])):
                all_pos.append(np.array([aux_array[0][count], \
                                    aux_array[1][count], aux_array[2][count]]))
            #all_pos.append([aux_array[np.argmax(aux_array[:, -1]), :]][:][0])
    """
    all_pos = Parallel(n_jobs=8, verbose=3)(delayed(process)(curr_data, labels, l, dist) \
                            for l in uni_labels)
    all_pos = list(all_pos)
    
    #print([n.shape for n in all_pos])
    
    return np.concatenate(all_pos, axis=0)


def append_circle_points(rad1, rad2, dx, N, y, x):
    
    #for y in range(int(N/2), int(rads[r+1] / dx + N/2)):
    points = [np.sqrt((x-N/2)**2 + (y-N/2)**2)][:][0]
    points = np.where((points < rad2) & (points >= rad1))

    ans_x = [x[points[-1]]][:][0]
    """
    all_pos_x.extend([ans_x][:][0])
    all_pos_x.extend([ans_x][:][0])
    all_pos_y.extend(list(y * np.ones((len(ans_x),))) + \
                    list((N - y) * np.ones((len(ans_x),))))
    """
    return list([ans_x][:][0]) + list([ans_x][:][0]), \
                        list(y * np.ones((len(ans_x),))) + \
                        list((N - y) * np.ones((len(ans_x),)))

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


def pad(arr, val):
    new_arr = np.zeros((arr.shape[0] + 2, arr.shape[1] + 2))
    new_arr[1:-1, 1:-1] = arr
    new_arr[:, 0] = val
    new_arr[:, -1] = val
    new_arr[0, :] = val
    new_arr[-1, :] = val

    return new_arr


# we suppose, that we have only 16 patches
def get_slice(arr, delta, n_slices, i):
    x, y = int(i / n_slices), i - n_slices * int(i / n_slices)
    if (x == n_slices - 1) and (y == n_slices - 1):
        return [arr[delta * x: , delta * y: ]][:][0]
    elif (x == n_slices - 1):
        return [arr[delta * x: , delta * y: delta * (y + 1)]][:][0]
    elif (y == n_slices - 1):
        return [arr[delta * x: delta * (x + 1), delta * y: ]][:][0]
    else:
        return [arr[delta * x: delta * (x + 1), delta * y: delta * (y + 1)]][:][0]


def distance_condition(xy, shift, delta, r_min):
    """
    distance = lambda x,y,x0,y0: np.sqrt((xy[0] - x0) ** 2 + \
                            (xy[1] - y0)**2)
    cond1 = distance(xy[0], xy[1], shift[1], shift[0]) > r_min
    cond2 = distance(xy[0], xy[1], delta + shift[1], shift[0]) > r_min
    cond3 = distance(xy[0], xy[1], shift[1], delta + shift[0]) > r_min
    cond4 = distance(xy[0], xy[1], ) > r_min
    if cond1 and cond2 and cond3 and cond4:
        return True
    else:
        return False
    """
    distance = lambda x,x0: np.abs(x - x0)
    cond1 = distance(xy[0], shift[1]) > r_min and distance(xy[1], shift[0]) > r_min
    cond2 = distance(xy[0], delta + shift[1]) > r_min and distance(xy[1], delta + shift[0]) > r_min
    cond1 = cond1 and cond2
    return cond1


def combine_cells(triplets, n_slices, r_min):
    delta = 36864. / n_slices

    orig_cell = [triplets[0]][:][0]
    curr_list = [triplets[0][:2]][:][0]
    curr_inten = [triplets[0][2]][:][0]
    # we assume current coords
    for i in range(1, n_slices * n_slices):
        shift = [delta * int(i / n_slices), delta * (i - n_slices * int(i / n_slices))]
        coords = [[triplets[i][0]][:][0], [triplets[i][1]][:][0]]
        coords = [coords[0] + shift[1], coords[1] + shift[0]]
        inten = [triplets[i][2]][:][0]
        new_subst = np.zeros((len(inten), ))
        old_subst = np.zeros((len(curr_inten), ))
        # looking for bording elements
        if len(coords[0]) == 0:
            continue
        if len(curr_list[0]) == 0:
            new_subst += 1
        else:
            local_KD = KDTree(np.concatenate([np.expand_dims(curr_list[0], -1), \
                                          np.expand_dims(curr_list[1], -1)], -1), leaf_size=1)
            ans_near = local_KD.query_radius((np.concatenate([np.expand_dims(coords[0], -1), \
                                          np.expand_dims(coords[1], -1)], -1)), r_min, \
                        count_only=False, return_distance=False)

            # substitution
            for a0, ans in enumerate(ans_near):
                is_take = [curr_inten[s] for s in ans if curr_inten[s] > inten[a0]]
                if len(is_take) == 0:
                    new_subst[a0] = 1
                    for board_elem in ans:
                        old_subst[board_elem] = 1
                
        curr_inten = curr_inten[old_subst==0]
        curr_list = [curr_list[0][old_subst==0], curr_list[1][old_subst==0]]

        curr_inten = np.concatenate([curr_inten, inten[new_subst==1]], -1)
        curr_list = [np.concatenate([curr_list[0], coords[0][new_subst==1]], -1), \
                    np.concatenate([curr_list[1], coords[1][new_subst==1]], -1)]
        print(i, " th slice of ", n_slices * n_slices)

    curr_list[0] -= (36864. / 2)
    curr_list[0] += (36864. / 2) / n_slices
    curr_list[1] -= (36864. / 2)
    curr_list[1] += (36864. / 2) / n_slices
    return curr_list + [curr_inten]


def combine_cells_simple(triplets, n_slices):
    delta = 36864. / n_slices

    orig_cell = [triplets[0]][:][0]
    curr_list = [triplets[0][:2]][:][0]
    curr_inten = [triplets[0][2]][:][0]

    for i in range(1, n_slices * n_slices):
        shift = [delta * int(i / n_slices), delta * (i - n_slices * int(i / n_slices))]
        coords = [[triplets[i][0]][:][0], [triplets[i][1]][:][0]]
        coords = [coords[0] + shift[1], coords[1] + shift[0]]
        inten = [triplets[i][2]][:][0]

        curr_inten = np.concatenate([curr_inten, inten], 0)
        curr_list = [np.concatenate([curr_list[0], coords[0]], -1), \
                    np.concatenate([curr_list[1], coords[1]], -1)]

    curr_list[0] -= (36864. / 2)
    curr_list[0] += (36864. / 2) / n_slices
    curr_list[1] -= (36864. / 2)
    curr_list[1] += (36864. / 2) / n_slices
    return curr_list + [curr_inten]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_slices", default=8, type=int)
    parser.add_argument("--n_threads", default=8, type=int)
    parser.add_argument("--begin_seed", default=1, type=int)
    parser.add_argument("--begin_dist", default=0, type=int)
    args = parser.parse_args()
    return vars(args)


def main():
    parsed = parse_args()
    n_slices = parsed["n_slices"]
    path_dir = "/home/abuzovkin/atmosphere/screens/36/scr"
    write_dir = "/home/abuzovkin/py_atm/big_data/iexpN"
    N = 36864
    dx = 3900. / N
    rads = np.array([0, 25, 45, 79, 112, 138, 160, 179, 197, 213, 228, 243, 270, \
                300, 330, 360, 390, 420, 450, 480, 510, 540, \
                570, 600, 630, 660, 700, 750, 800, 850, 900, 950, \
                1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400, \
                1500, 1600, 1700, 1800]) / dx
    rads = rads[::2]
    widths = np.array([1.076, 0.75, 0.537, 0.323])
    dists_slices = np.array([28000, 56000, 84000, 112000])[:-1]

    N1 = N
    #N1 = 100#3025
    x, y = np.meshgrid(np.arange(-np.floor(N1/2), np.ceil(N1/2))*dx, \
                    np.arange(-np.floor(N1/2), np.ceil(N1/2))*dx)
    #np.meshgrid(np.linspace(-dx * N1/2, dx * N1/2, N1), \
    #         np.linspace(-dx * N1/2, dx * N1/2, N1))
    r = np.sqrt(x * x + y * y)

    for ds0,ds in enumerate(dists_slices):

        if ds0 < parsed["begin_dist"]:
            continue

        func_inter = extract_an_vals(ds / 1000., np.max(r))
        if int(ds / 1000.) >= 84:
            r_mask = 3 * func_inter(r)
        else:
            r_mask = 2 * func_inter(r)

        for seed in range(parsed["begin_seed"], parsed["begin_seed"] + 1):
            if os.path.exists(os.path.join(write_dir, "_" + str(seed), \
                    str(seed)+"_"+str(ds)[:-3]+".csv")) is True:
                print(seed, ds, "exists")
                continue

            print("SEED", seed)
            #data = np.load("disk_small.npy")
            data = np.reshape(np.fromfile(os.path.join(path_dir, str(seed), \
             "psi_"+str(ds)+"_"+str(seed)), dtype=np.float64), [N1, N1])

            bool_mask = [np.where(data >= r_mask)][:][0]
                #[np.where(data >= -1)][:][0]#r_mask)][:][0]
            #screen_elems = process_cluster([x[bool_mask]][:][0], [y[bool_mask]][:][0], \
            #                    [data[bool_mask]][:][0], widths[ds0] / dx)

            #screen_elems = ringImax(r_mask, data, int(widths[ds0] / dx))
            #screen_elems = np.concatenate([np.expand_dims(n, axis=-1) for n in screen_elems], -1)
            if os.path.exists(os.path.join(write_dir, "_" + str(seed), \
                 str(seed)+"_"+str(ds)[:-3]+".csv")) == True:
                continue
            
            t1 = time.time()
            delta = int(data.shape[0] / n_slices)
            #cell_process = lambda i: ringImax(\
            #                pad(get_slice(r_mask, delta, i), 1.), 
            #                pad(get_slice(data, delta, i), 0.), int(widths[ds0] / dx))
            cell_process = lambda i: ringImax(\
                            get_slice(r_mask, delta, n_slices, i), 
                            get_slice(data, delta, n_slices, i), int(widths[ds0] / dx))
 
            screen_elems = Parallel(n_jobs=10, verbose=5)(delayed(cell_process)(j) \
                                for j in range(n_slices * n_slices))

            print(time.time() - t1)
            print(len(screen_elems),len(screen_elems[0]))
            screen_elems = combine_cells([screen_elems][:][0], n_slices, int(widths[ds0] / dx))
            #screen_elems = combine_cells_simple([screen_elems][:][0], n_slices, \
            #                    int(widths[ds0] / dx))

            if not os.path.exists(os.path.join(write_dir, "_" + str(seed))):
                    os.mkdir(os.path.join(write_dir, "_" + str(seed)))        
            pd.DataFrame(screen_elems).to_csv(os.path.join(write_dir, "_" + str(seed), \
                 str(seed)+"_"+str(ds)[:-3]+".csv"), \
                 header=None, index=None)            

            """
            for r in range(len(rads) - 1):
                temp_path = os.path.join(write_dir, "_" + str(seed), \
                     str(seed)+"_"+str(ds)[:-3]+"_"+str(int(rads[r]))+"_"+str(int(rads[r+1]))+".csv")
                if not os.path.exists(os.path.join(write_dir, "_" + str(seed))):
                    os.mkdir(os.path.join(write_dir, "_" + str(seed)))
                #if os.path.exists(temp_path):
                #    continue

                x = np.array(list(range(max(int(N/2 - rads[r+1]/dx), 0), \
                                        min(int(N/2 + rads[r+1]/dx) + 1, N - 1))))
                all_pos_x, all_pos_y = [], []

                all_init_pos = Parallel(n_jobs=8, verbose=3, batch_size=5)(\
                      delayed(append_circle_points)(rads[r], rads[r+1], dx, N, y, x) for \
                      y in range(int(N/2), int(rads[r+1] / dx + N/2)))
                all_pos_x, all_pos_y = [k[0] for k in all_init_pos], [k[1] for k in all_init_pos]
                all_pos_x = np.array([[item for sublist in all_pos_x for item in sublist]][:][0])
                all_pos_y = np.array([[item for sublist in all_pos_y for item in sublist]][:][0])
                all_pos_x = all_pos_x.astype(int)
                all_pos_y = all_pos_y.astype(int)
                del all_init_pos

                print("Collecting circle " + str(rads[r]) + " " + str(rads[r+1]) + " completed")
                print("Seed " + str(seed) + " dist " + str(ds))

                #print(type(all_pos_x[0]))
                #print(data[tuple(all_pos_x), tuple(all_pos_y)].shape)
                arr = [data[tuple(all_pos_x), tuple(all_pos_y)]][:][0]
                #arr_bool = [arr > np.max(arr) * 0.08][:][0]
                all_pos_x = np.array(all_pos_x)#[arr_bool]
                all_pos_y = np.array(all_pos_y)#[arr_bool]
                #arr = arr[arr_bool]

                thresh = extract_an_vals(ds / 1000, (rads[r] + rads[r+1]) / 2)
                centers = filtered_elems(all_pos_x, all_pos_y, widths[ds0] / dx, arr, \
                            thresh)
                if not os.path.exists(os.path.join(write_dir, "_" + str(seed))):
                    os.mkdir(os.path.join(write_dir, "_" + str(seed)))

                pd.DataFrame(centers).to_csv(os.path.join(write_dir, "_" + str(seed), \
                 str(seed)+"_"+str(ds)[:-3]+"_"+str(int(rads[r]))+"_"+str(int(rads[r+1]))+".csv"), \
                 header=None, index=None)
            """ 

if __name__=='__main__':
    main()
