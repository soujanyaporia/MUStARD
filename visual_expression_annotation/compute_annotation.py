#!/usr/bin/env python
import argparse

import krippendorff
import pandas as pd

import gesture_annotation


def parse_args():  # TODO: add help
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('output_filepath')
    parser.add_argument('--coarse-relabeling', action='store_true')
    return parser.parse_args()


def parse_dates(df):
    # Note that this function modifies df content and also returns it.

    df['SubmitTime'] = df['SubmitTime'].str.replace(r'P[SD]T', 'America/Los_Angeles', regex=True)
    df['SubmitTime'] = pd.to_datetime(df['SubmitTime'], format='%a %b %d %H:%M:%S %Z %Y')

    return df


def get_columns_by_answer_index(df):
    columns_by_index = {}
    i = 1
    while True:
        input_column_name = f'Input.video_url{i}'
        if input_column_name in df:
            columns_by_index[i] = [input_column_name]
            i += 1
        else:
            break
    for column_name in df:
        match = gesture_annotation.RE_ANSWER_WITH_INDEX_COLUMN.match(column_name)
        if match:
            i = int(match.group('index'))
            columns_by_index[i].append(column_name)
    return columns_by_index


def group_annotations(df):
    columns_by_answer_index = get_columns_by_answer_index(df)
    other_columns_to_keep = ['SubmitTime']

    # Concat the dataframes created from the selected columns,
    # which columns are renamed by removing the index suffix from it, supposing is always one digit.
    grouped_df = pd.concat((df[other_columns_to_keep + same_answer_index_columns].rename(columns=lambda name: name
                                                                                         if name == 'SubmitTime'
                                                                                         else name[:-1])
                            for same_answer_index_columns in columns_by_answer_index.values()), ignore_index=True)

    return grouped_df[grouped_df['Input.video_url'].notna()]


def remove_control_videos(df):
    return df[~df['Input.video_url'].str.startswith('http://lit.eecs.umich.edu/~annotations/sarcasm_2019/video_')]


def print_stats(df):
    print(f"Number of different instances: {df['Input.video_url'].nunique()}")
    print('')

    print(f"Number of (accepted) annotations: {len(df.index)}")
    annotations_per_video_series = df.groupby('Input.video_url').size()
    print(f"Min annotation number per video: {annotations_per_video_series.min()}")
    print(f"Max annotation number per video: {annotations_per_video_series.max()}")
    print(f"Average annotation number per video: {annotations_per_video_series.mean():.1f}")
    print('')


def compute_value_count_df(df, column_name):
    return df.pivot_table(values=[], index='Input.video_url', columns=column_name, aggfunc=len, fill_value=0)


def compute_value_count_df_dict(df):
    value_count_dict = {}
    for column_name in df:
        match = gesture_annotation.RE_ANSWER_WITHOUT_INDEX_COLUMN.match(column_name)
        if match:
            value_count_dict[match.group('feature')] = compute_value_count_df(df, column_name)
    return value_count_dict


def print_agreement(value_count_df_dict):
    for feature, value_count_df in value_count_df_dict.items():
        agreement_value = krippendorff.alpha(value_counts=value_count_df.values, level_of_measurement='nominal')
        print(f"Agreement for {feature}: {agreement_value:.4f}")
    print('')


def write_file_with_ties_to_break(df, filepath):
    annotated = []
    to_tiebreak = []
    for video_url, group_df in df.groupby('Input.video_url'):
        mode_df = group_df.sort_values('SubmitTime').head(3).drop('SubmitTime', axis=1).mode()
        if len(mode_df.index) > 1:
            mode_df = group_df.drop('SubmitTime', axis=1).mode()
        if len(mode_df.index) > 1:
            to_tiebreak.append(video_url)
        else:
            annotated.append(mode_df)

    print(f"{len(to_tiebreak)} instances need a tie break.")

    pd.concat(annotated).to_csv('annotated.csv', index=False)
    with open(filepath, 'w') as file:
        for video_url in to_tiebreak:
            file.write(f'{video_url}\n')


def main():
    args = parse_args()
    df = pd.read_csv(args.filepath)  # We don't use 'parse_dates' argument because it won't recognize "PST" or "PDT"
    #   as valid timezones.
    df = parse_dates(df)

    if args.coarse_relabeling:
        df = gesture_annotation.coarse_relabeling(df)

    grouped_df = group_annotations(df[df.AssignmentStatus == 'Approved'])
    grouped_df = remove_control_videos(grouped_df)
    print_stats(grouped_df)

    value_count_df_dict = compute_value_count_df_dict(grouped_df)
    print_agreement(value_count_df_dict)

    write_file_with_ties_to_break(grouped_df, args.output_filepath)


if __name__ == '__main__':
    main()
