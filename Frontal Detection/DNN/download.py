import urllib.request

# # Download the prototxt file
# url1 = 'https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt'
# urllib.request.urlretrieve(url1, 'deploy.prototxt')
#
# # Download the caffemodel file
# url2 = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel'
# urllib.request.urlretrieve(url2, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# url3 =  "https://download.01.org/opencv/2021/openvinotoolkit/2021.4/open_model_zoo/models_bin/3/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin"
# url4 =  "https://download.01.org/opencv/2021/openvinotoolkit/2021.4/open_model_zoo/models_bin/3/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"

url = 'https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/facial-landmarks-98-detection-0001.zip'
filename = 'file.zip'

urllib.request.urlretrieve(url, filename)

print('DONE')
