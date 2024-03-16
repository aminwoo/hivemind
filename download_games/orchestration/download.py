import os
import time
import requests
from requests.adapters import HTTPAdapter, Retry
from multiprocessing.pool import ThreadPool


def batch_download(urls, file_paths, num_threads=2):
    indices = list(range(len(urls)))
    counts = [len(urls)] * len(urls)
    params = list(zip(indices, counts, urls, file_paths))
    results = ThreadPool(num_threads).imap(download_url, params)
    for r in results:
        if r:
            print(*r)


def download_url(param):
    index, count, url, file_path = param
    num_digits = len(str(count))
    already_downloaded = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    if not already_downloaded:
        progress = "Downloading {0:0{2}}/{1:0{2}}:".format(index + 1, count, num_digits)
        s = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 443, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
        )
        s.mount('http://', HTTPAdapter(max_retries=retries))
        for i in range(2):
            try:
                r = s.get(
                    url=url,
                    stream=True,
                    timeout=60,
                    allow_redirects=True,
                    headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
                        "Accept-Encoding": "*",
                        "Connection": "keep-alive",
                        "referer": "https://www.google.com/",
                    },
                )
                if r.status_code == requests.codes.ok:
                    f = open(file_path, "wb")
                    for data in r:
                        f.write(data)
                    f.close()
                    return progress, url, file_path
                elif r.status_code == 404:
                    print("ERROR 404:", url)
                    f = open(file_path, "wb")
                    f.write(r.content)
                    f.close()
                    return progress, url, file_path
                elif r.status_code == 502:
                    pass
            except requests.exceptions.ReadTimeout:
                pass
            except requests.exceptions.ConnectionError:
                pass
            except requests.exceptions.ChunkedEncodingError:
                pass
            time.sleep(8)
    else:
        return None
    
