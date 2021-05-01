import os
import json
import subprocess
import argparse
import pandas as pd
from subprocess import Popen, PIPE

def download(csv_origin, csv_proc, output_dir, start=0, end=2**31):
    df = pd.read_csv(csv_origin, index_col=0)
    urls = {}
    for i in range(len(df)):
        if df['url'][i] not in urls and type(df['url'][i]) != float:
            urls[df['url'][i]] = 1
    urls = sorted(list(urls.keys()))
    os.makedirs(output_dir, exist_ok=True)
    unsups, url2id = [], {}
    log_failed = os.path.join(output_dir, 'failed.log.'+str(start))
    for i, url in enumerate(urls):
        if not (i >= start and i < end):
            continue
        cmd = ["youtube-dl", url , "--no-check-certificate", "--no-playlist", "--get-id", "--skip-download"]
        stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE).communicate()
        stderr = stderr.decode('utf-8').strip()
        if 'ERROR' in stderr:
            print(f"Unable to download {url}")
            unsups.append(url)
        else:
            yid = stdout.decode('utf-8').strip()
            cmd = ["youtube-dl", url , "--no-check-certificate", "--no-playlist", "-f", "best", "-o", os.path.join(output_dir, "%(id)s.%(ext)s")]
            stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE).communicate()
            stderr = stderr.decode('utf-8').strip()
            if 'ERROR' in stderr:
                print(f"Unable to download {url}")
                unsups.append(url)
            else:
                print(f"Downloaded {url} in {output_dir}")
                url2id[url] = yid
    open(log_failed, 'w').write('\n'.join(unsups))
    df_ = {key: [] for key in df.keys()}
    df_['yid'] = []
    for i in range(len(df)):
        if df['url'][i] in url2id:
            for key in df.keys():
                df_[key].append(df[key][i])
            df_['yid'].append(url2id[df['url'][i]])
    df_ = pd.DataFrame(df_)
    df_.to_csv(csv_proc)
    return
