from urllib import request

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36'}

def download_ucf101_dataset(path):
    try:
        req = request.Request('https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar', headers=headers)
        data = request.urlopen(req).read()
        with open(path, 'wb') as f:
            f.write(data)
            f.close()
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    download_ucf101_dataset('D:\\data\\zipped_datasets\\UCF101')