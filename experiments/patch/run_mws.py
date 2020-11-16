import numpy as np
import mutex_watershed as mws
import sys


def run_mws(affinities, offsets, stride, var=None, cons_aff=False):
    assert len(affinities) == len(offsets), "%s, %i" % (str(affinities.shape), len(offsets))
    affinities_ = np.require(affinities.copy(), requirements='C')

    vol_shape = affinities_.shape[2:]
    print(vol_shape)


    affinities_ = affinities_.ravel()

    if var is not None:
        var_ = var.ravel()
        # get ids and values, sorted by value
        tmp = np.array(list(zip(var_ ,affinities_,list(range(len(affinities_))))))
        tmp5 = tmp[tmp[:,0].argsort()[::-1]].transpose()
        _, sorted_edgesAff, sorted_edgesAffIds = tmp5
    elif cons_aff:
        affs = []
        reps = []
        print("splitting affs,reps")
        # for idx, a in enumerate(affinities_):
        #     if a < 0:
        #         reps.append((a, idx))
        #     elif a > 0:
        #         affs.append((a, idx))
        affs = zip(affinities_[affinities_>0], np.where(affinities_>0)[0])
        affs = np.array(list(affs))
        print(affs.shape)
        print("got affs")
        reps = zip(-affinities_[affinities_<0], np.where(affinities_<0)[0])
        # reps = zip(-affinities_[affinities_<=0], np.where(affinities_<=0)[0])
        reps = np.array(list(reps))
        print(reps.shape)
        print("done")
        print(len(reps), len(affs))
        print("sorting")
        affs_t = affs[affs[:,0].argsort()[::-1]].transpose()
        sorted_edgesAff, sorted_edgesAffIds = affs_t
        reps_t = reps[reps[:,0].argsort()[::-1]].transpose()
        sorted_edgesRep, sorted_edgesRepIds = reps_t
        print("done")
    else:
        print("getting reps")
        repulsions_ = 1.0 - affinities_
        repulsions_ = repulsions_.ravel()
        print("sorting affs")
        tmp = np.array(list(zip(affinities_,list(range(len(affinities_))))))
        tmp5 = tmp[tmp[:,0].argsort()[::-1]].transpose()
        sorted_edgesAff, sorted_edgesAffIds = tmp5
        print("sorting reps")
        tmp = np.array(list(zip(repulsions_,list(range(len(repulsions_))))))
        tmp5 = tmp[tmp[:,0].argsort()[::-1]].transpose()
        sorted_edgesRep, sorted_edgesRepIds = tmp5
    print(sorted_edgesAff.shape, sorted_edgesRep.shape)
    var = None

    # run the mst watershed
    print("starting mws")
    seperating_channel = 3 # ignored
    mst = mws.MutexWatershed(np.array(vol_shape),
                             offsets,
                             seperating_channel,
                             stride)
    mst.repulsive_ucc_mst_cut(sorted_edgesAff, sorted_edgesAffIds,
                              sorted_edgesRep, sorted_edgesRepIds,
                              len(sorted_edgesAff),
                              len(sorted_edgesRep),
                              0, var is not None)
    segmentation = mst.get_flat_label_image().reshape(vol_shape)
    return segmentation
