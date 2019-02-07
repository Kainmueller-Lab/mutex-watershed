import numpy as np
import mutex_watershed as mws
import sys


def run_mws(affinities,
            offsets, stride):
    assert len(affinities) == len(offsets), "%s, %i" % (str(affinities.shape), len(offsets))
    affinities_ = np.require(affinities.copy(), requirements='C')
    # affinities_ = affinities_[0:49,:,:]
    # affinities_ = affinities_.reshape((25,25,476,476))
    # affinities_ = affinities_[9:16,9:16,:,:]
    # affinities_ = affinities_.reshape((-1,476,476))
    # affinities_ = np.require(affinities_.copy(), requirements='C')

    vol_shape = affinities_.shape[1:]

    repulsions_ = 1.0 - affinities_
    affinities_ = affinities_.ravel()
    repulsions_ = repulsions_.ravel()

    # get ids and values, sorted by value
    tmp = np.array(list(zip(affinities_,list(range(len(affinities_))))))
    tmp5 = tmp[tmp[:,0].argsort()[::-1]].transpose()
    sorted_edgesAff, sorted_edgesAffIds = tmp5
    # print(sorted_edgesAffIds[:10], sorted_edgesAff[:10])
    tmp = np.array(list(zip(repulsions_,list(range(len(repulsions_))))))
    tmp5 = tmp[tmp[:,0].argsort()[::-1]].transpose()
    sorted_edgesRep, sorted_edgesRepIds = tmp5
    # print(sorted_edgesRepIds[:10], sorted_edgesRep[:10])

    # run the mst watershed
    seperating_channel = 3 # ignored
    mst = mws.MutexWatershed(np.array(vol_shape),
                             offsets,
                             seperating_channel,
                             stride)
    mst.repulsive_ucc_mst_cut(sorted_edgesAff, sorted_edgesAffIds,
                              sorted_edgesRep, sorted_edgesRepIds,
                              len(sorted_edgesRep),
                              0)
    segmentation = mst.get_flat_label_image().reshape(vol_shape)
    return segmentation
