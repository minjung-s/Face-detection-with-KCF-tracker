import detectors

detector_maps = {
    'haar' : detectors.HaarCascadeDetector(),
    'haar2' : detectors.HaarCascadeDetector(),
    'LBP' : detectors.LBPCascadeDetector(),
    'LBP2' : detectors.LBPCascadeDetector(),
    'OpenCVDNN' : detectors.OpenCVDNNDetector(),
    'HoG' : detectors.HoGSVMDetector(),
    'DlibDNN' : detectors.DlibDNNDetector(),
    'MTCNN' : detectors.MTCNNDetector(),
    # 'RMTCNN' : detectors.RMTCNNDetector(),
}

def get_detector(name):
    if name not in detector_maps:
        raise ValueError('Name of detector unknown %s' % name)

    return detector_maps[name]
