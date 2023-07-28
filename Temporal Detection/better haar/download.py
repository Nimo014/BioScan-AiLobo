import urllib.request

url = 'https://github.com/adobe/SimpleSensor/blob/master/simplesensor/collection_modules/demographic_camera/classifiers/haarcascades/haarcascade_mcs_leftear.xml'
filename = 'cascade_left_ear.xml'

urllib.request.urlretrieve(url, filename)

print('Done')
