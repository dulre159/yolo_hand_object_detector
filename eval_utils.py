import os
import numpy as np

import config
from voc_eval import voc_eval, voc_eval_hand
import pickle

def _do_python_eval(datapath, localdatapath, imagenames):
    annopath = os.path.join(
        datapath,
        'Annotations',
        '{:s}.xml')
    # imagesetfile = os.path.join(
    #     datapath,
    #     'ImageSets',
    #     'Main',
    #     'test.txt')
    # imagesetfile = os.path.join(
    #     config.DATASET,
    #     'test.csv')
    cachedir = os.path.join(localdatapath, 'annotations_cache_voc_2007')
    aps = []
    # print('VOC07 metric MAP test\n')
    apsd = {'targetobject':0, 'hand':0, 'handside':0, 'handstate':0, 'objectbbox':0, 'all':0, 'mean':0}
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    for i, cls in enumerate(['__background__','targetobject','hand']):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(localdatapath).format(cls)
        rec, prec, ap = voc_eval(filename, annopath, imagenames, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        # print('AP for {} = {:.4f}\n'.format(cls, ap))
        apsd[cls] = ap

        if cls == 'hand':
            filename = _get_voc_results_file_template(localdatapath)  # .format(cls)
            for constraint in ['handstate', 'handside', 'objectbbox', 'all']:
                rec, prec, ap = voc_eval_hand(filename, annopath, imagenames, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric, constraint=constraint)
                apsd[constraint] = ap
                # print('AP for {} + {} = {:.4f}\n'.format(cls, constraint, ap))
                # with open(os.path.join(output_dir, cls + f'_pr_{constraint}.pkl'), 'wb') as f:
                #     pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    # print('Mean AP = {:.4f}'.format(np.mean(aps)))
    apsd['mean'] = np.mean(aps)
    # print('~~~~~~~~')
    # print('Results:')
    # for ap in aps:
        # print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    return apsd

def evaluate_detections(all_boxes, datapath, localdatapath, imagenames):
    _write_voc_results_file(all_boxes, localdatapath, imagenames)
    return _do_python_eval(datapath, localdatapath, imagenames)

def _write_voc_results_file(all_boxes,localdatapath, imagenames):
    for cls_ind, cls in enumerate(['__background__','targetobject','hand']):
        if cls == '__background__':
            continue
        # print('Writing {} VOC results file'.format(cls))
        filename = _get_voc_results_file_template(localdatapath).format(cls)
        with open(filename, 'wt') as f:
            # image_index = _load_image_set_index(config.DATASET, "test")
            image_index = imagenames
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.
                            format(index, dets[k, 4],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1,
                                   int(dets[k, 5]), dets[k, 6], dets[k, 7], dets[k, 8], dets[k, 9], dets[k, 10]))

def _load_image_set_index(datapath, image_set="test"):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    # image_set_file = os.path.join(datapath, 'ImageSets', 'Main', image_set + '.txt')
    image_set_file = os.path.join(datapath, image_set + '.csv')
    assert os.path.exists(image_set_file), \
        'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [os.path.splitext(x.strip().split(",")[0])[0] for x in f.readlines()]
    return image_index

# def _get_default_path(self):
#     """
#     Return the default path where PASCAL VOC is expected to be installed.
#     """
#     default_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year +'_handobj_100K')
#     print()
#     print(f'--------> dataset path = {default_path}')
#     print()
#     return default_path

def _get_voc_results_file_template(localdatapath):
    # DATASETFOLDER/detection_results_voc_2007/det_test_aeroplane.txt
    filename = 'det_test_{:s}.txt'
    filedir = os.path.join(localdatapath, 'detection_results_voc_2007')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path