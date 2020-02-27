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
    for l in labels:
        cnt = np.count_nonzero(seg==l)
        if cnt == 1:
            seg[seg==l] = 0

    return seg

@timer
def mws_result(affinities, offsets, strides):
    from run_mws import run_mws
    return run_mws(affinities, offsets, strides)

def experiment(aff_path, aff_key, result_folder):
    # affinity offsets
    offsets = []
    ln = 20
    for i in range(-ln, ln+1):
        for j in range(-ln, ln+1):
            offsets.append([i, j])
    print(offsets, len(offsets))
    
    if aff_path.endswith(".zarr"):
        zf = zarr.open(aff_path, mode="r")
        affs = zf[aff_key][:]
    elif aff_path.endswith(".hdf"):
        with h5py.File(aff_path) as f:
            affs = f[aff_key][:]
    else:
        raise NotImplementedError

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    strides = np.array([1., 1., 1.])
    print("Computing mutex watershed segmentation ...")
    mws_seg, t_mws = mws_result(affs, offsets, strides)
    print("... finished in  %f s" % t_mws)

    # output of mws: each background pixel own instance
    # relabel st background = 0
    mws_seg, fixed = relabel(mws_seg)
    if not fixed:
        mws_seg = remove_tiny_inst(mws_seg)
        mws_seg, _ = relabel(mws_seg, keepZero=True)
    
    sample_name = os.path.basename(aff_path).split(".")[0]
    writeHDF5(mws_seg, os.path.join(result_folder, sample_name + '.hdf'),
              'instances', compression='gzip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('aff_path', type=str, help='path to affinities (hdf5 file)')
    parser.add_argument('aff_key', type=str, help='path to affinity dataset in hdf5 file')
    parser.add_argument('result_folder', type=str,
                        help='folder to save result segmentations as hdf5')

    args = parser.parse_args()
    aff_path = args.aff_path
    aff_key = args.aff_key
    assert os.path.exists(aff_path), aff_path

    experiment(aff_path, aff_key, args.result_folder)

if __name__ == '__main__':
    main()
