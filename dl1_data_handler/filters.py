import numpy as np

#################
# Event Filters #
#################


def event_intensity_filter(reader, file, i_min=-np.inf, i_max=np.inf):
    """
    Prototype of Filter events on intensity (in pe)
    Parameters
    ----------
    reader (DL1DataReader) : the reader to filter
    file
    i_min
    i_max

    Returns
    -------
    the filtered indices
    """
    # TODO define a physically correct strategy
    tel_types = [reader.tel_type] if reader.mode in ['mono', 'stereo'] else list(reader.selected_telescopes)
    total_intensity = np.zeros(len(file.root.Events))
    for tel_type in tel_types:
        indices = file.root.Events[:][tel_type + '_indices']
        images = file.root['Images']._f_get_child(tel_type)[:]['charge']
        images = images[indices]
        total_intensity += images.sum(axis=(1, 2))
    mask1 = i_min < total_intensity
    mask2 = total_intensity < i_max
    return set(np.arange(len(file.root.Events))[mask1 & mask2])


#################
# Image Filters #
#################


def image_intensity_filter(reader, images, i_min=-np.inf, i_max=np.inf):
    """
    Filter images on intensity (in pe)
    Parameters
    ----------
    reader (DL1DataReader) : the reader to filter
    images
    i_min
    i_max

    Returns
    -------
    mask (Array of bool)
    """

    amps = images.sum(axis=-1)
    mask1 = i_min < amps
    mask2 = amps < i_max
    return mask1 & mask2


def image_intensity_after_cleaning_filter(reader, images, i_min=-np.inf, i_max=np.inf, **opts):
    """
    Filter images on intensity (in pe) after cleaning

    Parameters
    ----------
    reader: `DL1DataReader`
    images
    options for image cleaning
    i_min
    i_max

    Returns
    -------
    mask (Array of bool)
    """

    try:
        from ctapipe.image import cleaning
    except ImportError:
        raise ImportError("The `ctapipe.image.cleaning` python module is required to perform cleaning operation")
    try:
        from ctapipe.instrument.camera import CameraGeometry
    except ImportError:
        raise ImportError("The `ctapipe.instrument.CameraGeometry` python module is required to perform cleaning operation")

    geom = CameraGeometry.from_name(reader.tel_type.split('_')[-1])

    def int_after_clean(img):
        cleanmask = cleaning.tailcuts_clean(geom, img, **opts)
        clean = img.copy()
        clean[~cleanmask] = 0.0
        amps = np.sum(clean)
        return (i_min < amps) & (amps < i_max)

    int_mask = np.apply_along_axis(int_after_clean, 1, images)
    return int_mask


def image_cleaning_filter(reader, images, **opts):
    """
    Filter images according to a cleaning operation.

    Parameters
    ----------
    reader: `DL1DataReader`
    images
    options for image cleaning

    Returns
    -------
    mask (Array of bool)
    """

    try:
        from ctapipe.image import cleaning
    except ImportError:
        raise ImportError("The `ctapipe.image.cleaning` python module is required to perform cleaning operation")
    try:
        from ctapipe.instrument.camera import CameraGeometry
    except ImportError:
        raise ImportError("The `ctapipe.instrument.CameraGeometry` python module is required to perform cleaning operation")

    geom = CameraGeometry.from_name(reader.tel_type.split('_')[-1])

    def clean(img):
        return cleaning.tailcuts_clean(geom, img, **opts)

    clean_mask = np.apply_along_axis(clean, 1, images)
    return clean_mask.any(axis=1)


def leakage_filter(reader, images, leakage_value=1.0, leakage_number=2, **opts):
    """
    Filter images on leakage
    Comment: An image cleaning filter is applied by default.

    Parameters
    ----------
    reader: `DL1DataReader`
    images
    options for image cleaning
    leakage_value
    leakage_number

    Returns
    -------
    mask (Array of bool)
    """

    try:
        from ctapipe.image import cleaning, leakage
    except ImportError:
        raise ImportError("The `ctapipe.image.cleaning` and/or `ctapipe.image.leakage` python module is required to perform leakage operation")
    try:
        from ctapipe.instrument.camera import CameraGeometry
    except ImportError:
        raise ImportError("The `ctapipe.instrument.CameraGeometry` python module is required to perform leakage operation")

    if leakage_number not in [1,2]:
        raise ValueError("The leakage_number is {}. Valid options are 1 or 2.".format(leakage_number))

    geom = CameraGeometry.from_name(reader.tel_type.split('_')[-1])

    def leak(img):
        cleanmask = cleaning.tailcuts_clean(geom, img, **opts)
        mask = False
        if any(cleanmask):
            leakage_values = leakage(geom, img, cleanmask)
            if hasattr(leakage_values, 'leakage{}_intensity'.format(leakage_number)):
                mask = leakage_values['leakage{}_intensity'.format(leakage_number)] <= leakage_value
            elif hasattr(leakage_values, 'intensity_width_{}'.format(leakage_number)):
                mask = leakage_values['intensity_width_{}'.format(leakage_number)] <= leakage_value
        return mask
    leakage_mask = np.apply_along_axis(leak, 1, images)
    return leakage_mask

