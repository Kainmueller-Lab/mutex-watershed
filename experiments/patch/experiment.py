import zarr
import os
import time
import argparse
import numpy as np
import h5py
import zarr


def writeHDF5(data, path, key, **kwargs):
    with h5py.File(path, 'w') as f:
        f.create_dataset(key, data=data, **kwargs)

def timer(func):
    def func_wrapper(*args, **kwargs):
        t0 = time.time()
        segmentation = func(*args, **kwargs)
        return segmentation, time.time() - t0
    return func_wrapper

def replace(array, old_values, new_values, return_mapping=False):
    """replace multiple values at once in array"""
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    # builds new array with shape=array.shape
    # takes values from array as index into values_map
    # and the value from values_map is put into array
    if return_mapping:
        return values_map[array], values_map
    else:
        return values_map[array]

def relabel(seg, keepZero=False):
    """recompute seg labeling
    using new ids starting at zero
    """
    print(np.max(seg))
    labels = sorted(list(np.unique(seg)))
    if keepZero and 0 in labels:
        labels.remove(0)
    print(len(labels))
    if len(labels) < 2**16:
        tp = np.uint16
        fixed = True
        print("Segmentation contains {} distinct labels, can be expressed in int16.".format(len(labels)))
    else:
        tp = np.uint64
        fixed = False
        print("Segmentation contains {} distinct labels, can not be expressed in int16.".format(len(labels)))

    old_values = np.array(labels)
    new_values = np.arange(1, len(labels) + 1, dtype=tp)

    seg = replace(seg, old_values, new_values)

    return seg, fixed

def remove_tiny_inst(seg):
    labels = list(np.unique(seg))
    if 0 in labels:
        mx = max(labels)
        seg[seg==0] = mx+1
        labels.remove(0)
        labels.append(mx+1)
    print("cleaning up background...")
    ls, cs = np.unique(seg, return_counts=True)
    for l, c in zip(ls, cs):
        if c < 2:
            seg[seg==l] = 0

    return seg

@timer
def mws_result(affinities, offsets, strides, var=None, cons_aff=False):
    from run_mws import run_mws
    return run_mws(affinities, offsets, strides, var=var, cons_aff=cons_aff)

def experiment(aff_path, aff_key, PS, result_folder, var_key=None, cons_aff=False):
    # affinity offsets
    offsets = []
    if cons_aff:
        NS = PS*2
        i = PS
        for j in range(NS//2, NS):
            offsets.append([i-PS, j-PS])
        for i in range(NS//2+1, NS):
            for j in range(0, NS):
                offsets.append([i-PS, j-PS])
    else:
        ln = PS//2
        for i in range(-ln, ln+1):
            for j in range(-ln, ln+1):
                offsets.append([i, j])
    print(offsets, len(offsets))

    if aff_path.endswith(".hdf"):
        with h5py.File(aff_path) as f:
            affs = f[aff_key][:]
    elif aff_path.endswith(".zarr"):
        f = zarr.open(aff_path, 'r')
        affs = f[aff_key][:]
    elif aff_path.endswith(".npy"):
        affs = np.load(aff_path)
    else:
        raise NotImplementedError
    print(affs.shape)
    if cons_aff:
        affs = np.squeeze(affs)
        tmp = affs[PS-1,PS:NS,...]
        for i in range(PS, affs.shape[0]-1):
            tmp = np.concatenate([tmp, affs[i,:NS,...]], axis=0)
        print(tmp.shape)
        affs = tmp
        affs.shape = (affs.shape[0], 1, affs.shape[1], affs.shape[2])
        # just for visualization
        # affs = np.reshape(affs, (NS*NS,)+ affs.shape[3:])
        # affs = affs[(PS+(PS-1)*NS+1):(NS-1)*NS]
        # with h5py.File(aff_path[:-4] + "test.hdf", 'w') as f:
        #     f.create_dataset(
        #         'volumes/cons',
        #         data=affs,
        #         compression='gzip')
    else:
        if affs.shape[1] != 1:
            affs.shape = (affs.shape[0], 1, affs.shape[1], affs.shape[2])
    var = None
    if var_key is not None:
        with h5py.File(aff_path) as f:
            var = f[var_key][:]

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    strides = np.array([1., 1., 1.])
    print("Computing mutex watershed segmentation ...")
    mws_seg, t_mws = mws_result(affs, offsets, strides, var=var, cons_aff=cons_aff)
    print("... finished in  %f s" % t_mws)

    # output of mws: each background pixel own instance
    # relabel st background = 0
    mws_seg, fixed = relabel(mws_seg)
    # if not fixed:
    mws_seg = remove_tiny_inst(mws_seg)
    mws_seg, _ = relabel(mws_seg, keepZero=True)

    sample_name = os.path.basename(aff_path).split(".")[0]
    writeHDF5(mws_seg, os.path.join(result_folder, sample_name + '.hdf'),
              'instances', compression='gzip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, help='path to affinities (hdf5 file)', required=True)
    parser.add_argument('--aff_key', type=str, help='path to affinity dataset in hdf5 file', required=True)
    parser.add_argument('--var_key', type=str, help='old, path to variance of affinities dataset in hdf5 file', required=False)
    parser.add_argument("--cons", action="store_true",
                        help='consensus affinities (different offsets)')
    parser.add_argument('--result_folder', type=str,
                        help='folder to save result segmentations as hdf5', required=True)
    parser.add_argument('--patch_size', type=int,
                        help='size of used patches', required=True)

    args = parser.parse_args()
    aff_path = args.aff_path
    aff_key = args.aff_key
    var_key = args.var_key
    assert os.path.exists(aff_path), aff_path

    experiment(aff_path, aff_key, args.patch_size, args.result_folder, var_key=var_key, cons_aff=args.cons)

if __name__ == '__main__':
    main()
