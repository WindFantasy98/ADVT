from urllib import request

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36'}

def download_hmdb51_dataset(path):
    try:
        req = request.Request('http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar', headers=headers)
        data = request.urlopen(req).read()
        with open(path, 'wb') as f:
            f.write(data)
            f.close()
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    download_hmdb51_dataset('D:\\data\\zipped_datasets\\HDMI51')