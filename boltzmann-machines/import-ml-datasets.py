# Run this program to download the ml-1m and ml-100k datasets

import requests, zipfile
from pathlib import Path

url_100k = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
url_1m = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

# ml-1m
if (Path('./ml-1m').exists()):
    print('ml-1m dataset available and extracted.')
else:
    print('ml-1m dataset not available.\nDownloading...')

    with open(Path('./ml-1m.zip'), mode='wb') as f:
        f.write(requests.get(url_1m).content)

    print('Extracting...')
    with zipfile.ZipFile(file=Path('./ml-1m.zip'), mode='r') as f:
        f.extractall(Path('./ml-1m/'))

if (Path('./ml-1m.zip').exists()):
    Path('./ml-1m.zip').unlink()
    print('Deleted ml-1m.zip file')

# ml-100k
if (Path('./ml-100k').exists()):
    print('ml-100k dataset available and extracted.')
else:
	print('ml-100k dataset not available.\nDownloading...')

	with open(Path('./ml-100k.zip'), mode='wb') as f:
		f.write(requests.get(url_100k).content)
	
	print('Extracting...')
	with zipfile.ZipFile(file=Path('./ml-100k.zip'), mode='r') as f:
		f.extractall(Path('./ml-100k/'))

if (Path('./ml-100k.zip').exists()):
    Path('./ml-100k.zip').unlink()
    print('Deleted ml-100k.zip file')