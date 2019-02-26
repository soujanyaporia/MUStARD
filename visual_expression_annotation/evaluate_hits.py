#!/usr/bin/env python
import argparse
import json
import os

import pandas as pd

import gesture_annotation


def parse_args():  # TODO: descriptions
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument(metavar='control_video_answers_file', dest='control_video_answers_filepath', help="TODO")
    parser.add_argument(metavar='batch_results_file', dest='batch_results_filepath', help="TODO")
    return parser.parse_args()


def get_out_filepath(in_filepath):
    dir_path, in_filename = os.path.split(in_filepath)
    in_filename_without_ext, ext = os.path.splitext(in_filename)
    return os.path.join(dir_path, f'{in_filename_without_ext}_revised{ext}')


def get_control_video_answers(filepath):
    with open(filepath) as file:
        return json.load(file)


def main():
    args = parse_args()

    df = pd.read_csv(args.batch_results_filepath)

    control_videos_answers_dict = get_control_video_answers(args.control_video_answers_filepath)

    hit_is_complete_series = True
    hit_passes_control_series = True
    for column in df:
        match = gesture_annotation.RE_ANSWER_WITH_INDEX_COLUMN.match(column)
        if match:
            feature = match.group('feature')
            i = int(match.group('index'))
            hit_is_complete_series &= df[f'Input.video_url{i}'].isna() | df[column].notna()

    #     videoURLs = line[27: 32]
    #     annotation = line[32:]
    #
    #     followInstr = True
    #     for i in range(args.videos_per_hit):
    #         videoURL = videoURLs[i]
    #         videoName = videoURL[52:]
    #         if videoName in control_videos_answers_dict:
    #             correctAnnotationList = control_videos_answers_dict[videoName]
    #             print(videoName)
    #
    #             for j in range(len(correctAnnotationList)):
    #                 correctAnnotation = correctAnnotationList[j]
    #                 print("correct annotation", correctAnnotation)
    #                 print("amt", annotation[j * 5 + i])
    #                 if correctAnnotation != annotation[j * 5 + i]:
    #                     followInstr = False
    #                     break
    #     if not followInstr:
    #         outfileList.writerow(line + ['', 'Did not follow instructions'])
    #         continue

    out_filepath = get_out_filepath(args.batch_results_filepath)
    df.to_csv(out_filepath, index=False)


if __name__ == '__main__':
    main()
