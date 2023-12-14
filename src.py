import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.segmentation as seg
import torch
from skimage import color, exposure
from skimage.io import imread
from skimage.transform import resize

def cl(val):
    ival = int(val)
    if ival==255:
        return 254,True
    if ival==0:
        return 0,True
    else:
        return ival,0


def progress_bar(cur, lim):
    """
    prints cur/lim*100%
    """
    v = int(round(cur / float(lim) * 100))
    sys.stdout.write('\r' + str(v) + '%')
    if v == 100: sys.stdout.write('\n')


def load(root='PH2Dataset', img_dir='PH2 Dataset images'):
    """
    Загрузка изображений из датасета.
    :param root: Путь к директории с датасетом
    :param img_dir: Имя директории с картинками в папке
    :return: X, Y
    """
    images = []
    lesions = []

    for root, dirs, files in os.walk(os.path.join(root, img_dir)):
        if root.endswith('_Dermoscopic_Image'):
            images.append(imread(os.path.join(root, files[0])))
        if root.endswith('_lesion'):
            lesions.append(imread(os.path.join(root, files[0])))

    size = (256, 256)
    X = [resize(x, size, mode='constant', anti_aliasing=True, ) for x in images]
    Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]

    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)
    return X, Y


def iou_np(pred, eth, thresholded=False) -> float:
    """
    IoU = (eth && pred)/(eth || pred)
    :param pred: Prediction
    :param eth: Target
    :return: IoU: float
    """
    intersection = np.logical_and(pred, eth)
    union = np.logical_or(pred, eth)
    iou = np.count_nonzero(intersection) / np.count_nonzero(union)
    if thresholded:
        _thresholded = torch.clamp(20 * (iou - 0.5), 0,
                                   10).ceil() / 10
        iou = _thresholded
    return float(iou)


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap=cmap)
    ax.axis('off')
    return fig, ax


def image_show_cl(image, nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image)
    ax.axis('off')
    return fig, ax


def plot_4(imgs: list):
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.imshow(imgs[0])
    # plt.imshow(X[i])

    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.imshow(imgs[1])

    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.imshow(imgs[2])

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.imshow(imgs[3])


def scale_contrast(image, rg: list):
    """
    Изменяет диапазон значений цветов в картинке. Используетяс чтобы увеличить контрастность для лучшей сегментации
    :param image:
    :param rg: range: (a,b)
    :return: Rescaled image
    """
    rg = tuple(np.array(rg))
    percentiles = np.percentile(image, rg)
    scaled = exposure.rescale_intensity(image,
                                        in_range=tuple(percentiles))
    return scaled


def get_optic_mask(file_name="mask"):
    """
    Возвращает маску, обрезающую кромку изображений
    :param file_name:
    :return:
    """
    mask = np.loadtxt(file_name).astype(np.bool)
    return mask


def segmentize(image: np.ndarray, sigma = .6, scale = 900, min_size = 1000, init_pcc = [0,100],div=5):
    """
    Сегментирует изображение. Предварительно значения пикселей масштабируются атносительно матожидения, чтобы увеличить контраст и сделать дефект на коже более различимым
    :return: Сегментированное и предобработанное(контрастное) изображение
    """
    mask = get_optic_mask()
    _image = image.copy()
    # _eth = eth.copy()
    img = image.copy()

    image_gray = skimage.color.rgb2gray(_image)
    im_flat = image_gray[mask]

    init_percentiles = np.percentile(im_flat, init_pcc)

    band = image_gray[np.logical_and(image_gray>=init_percentiles[0],image_gray<=init_percentiles[1])]
    M = band.mean()

    S = np.std(band) / div

    im_gray = skimage.color.rgb2gray(_image)
    im_gray[np.bitwise_not(mask)] = 0
    percentiles = np.array([M - S, M + S])
    scaled = exposure.rescale_intensity(im_gray,
                                        in_range=tuple(percentiles))

    res = seg.felzenszwalb(scaled, sigma=sigma, scale=scale, min_size=min_size)

    return res, scaled


def predict(img, segmented_img: np.ndarray, tp_size_th=500,sigma=5, thh=.5) -> (int, np.ndarray):
    """
    По сегментированному и исходному изображению определить на каком обнаруживаемые болячки
    Здесь фильтруются области меньше порогового  значения. Из оставшихся выбирается самая тёмная
    :param img: Исходное изображение
    :param segmented_img: Сегментированное изображение
    :param tp_size_th: орог размера
    :return: pred, n_regions
    """
    mask = get_optic_mask()
    masked = segmented_img.copy()

    masked[np.bitwise_not(mask)] = -1
    types = {tp: np.count_nonzero(masked == tp) for tp in np.unique(masked[masked >= 0])}

    n_regions = len(types)

    smol_types = [tp for tp, cnt in types.items() if cnt <= tp_size_th]

    bigs = masked
    for tp in smol_types:
        bigs[bigs == tp] = -1
    types = {tp: [np.count_nonzero(bigs == tp),np.sum(img[bigs == tp])] for tp in np.unique(bigs) if tp >= 0}
    vals = np.vstack([np.array(it) for it in types.values()])

    arr_types = np.vstack([list(types),vals[:,0],vals[:,1], (np.divide(vals[:,1],vals[:,0]))]).transpose()

    darkest_index = np.argmin(arr_types[:,3])
    darkest = arr_types[darkest_index]
    darkest_area = masked==(int(darkest[0]))
    blured = skimage.filters.gaussian(darkest_area, sigma=(sigma, sigma))
    area = blured>thh
    area[np.bitwise_not(mask)]=0
    pred =area
    return pred, n_regions


def process_(X,Y,sigma_seg=.6, scale = 1000, min_size = 1000, init_pcc=(0,100),sigma_pred=5, thh_pred=.5,lim=None,div=5)-> (np.ndarray,np.ndarray):
    """
    Принимает датасет и обрабатывает его весь или частично(lim).
    :return: Возвращает список маетрик IoU по всем изображениеям, а так же итоговые оценки(Изображения с предсказанными деффектами)
    """
    ious=[]
    preds=[]
    if lim is None: lim=len(X)
    for i in range(lim):
        progress_bar(i,lim)
        _image:np.ndarray = X[i]
        _eth:np.ndarray = Y[i]

        segmented,processed = segmentize(_image,sigma=sigma_seg,scale=scale,min_size=min_size,init_pcc=init_pcc,div=div)
        # src.image_show_cl(segmented)
        pred, n_regions = predict(_image,segmented, tp_size_th=1000,sigma=sigma_pred,thh=thh_pred)
        iou = iou_np(pred,_eth)

        # src.plot_4([pred,_eth,segmented,_image])
        ious.append(iou)
        preds.append(pred)
    return np.array(ious), np.array(preds)