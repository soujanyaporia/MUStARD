#!/usr/bin/env python
import argparse
import csv
import random
import sys

import pandas as pd


URL_BASE_VIDEOS = 'http://lit.eecs.umich.edu/~annotations/sarcasm/utterances/'
URL_BASE_CONTROL_VIDEOS = 'http://lit.eecs.umich.edu/~annotations/sarcasm_2019/'


def parse_args():  # TODO: add help
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('control_file')
    parser.add_argument('control_gesture')
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.file)

    ds = df.sample(frac=1)

    ids = ds.index.tolist()

    df2 = pd.read_csv(args.control_file)
    ds2 = df2.sample(frac=1)
    ds2 = df2.loc[df2['gesture'] == args.control_gesture]
    control = ds2.index.tolist()
    n_videos = 4
    indexes = [ids[i * n_videos:(i + 1) * n_videos] for i in range((len(ids) + n_videos - 1) // n_videos)]

    writer = csv.writer(sys.stdout)
    writer.writerow(['video_url1', 'video_url2', 'video_url3', 'video_url4', 'video_url5'])
    for set_i in indexes:
        c = random.choice(control)
        urls = [URL_BASE_VIDEOS + str(id_) + '.mp4' for id_ in ds.loc[set_i, 'id']]
        urls.append(URL_BASE_CONTROL_VIDEOS + ds2.loc[c, 'id'])
        random.shuffle(urls)
        writer.writerow(urls)


if __name__ == '__main__':
    main()
