import re

RE_ANSWER_WITH_INDEX_COLUMN = re.compile(r'^Answer\.(?P<feature>[A-Za-z]+)(?P<index>\d)$')
RE_ANSWER_WITHOUT_INDEX_COLUMN = re.compile(r'^Answer\.(?P<feature>[A-Za-z]+)$')

ANNOTATION_OPTIONS_BY_TYPE_AND_FEATURE_DICT = {
    'eye': {
        'EyeMovement': ['Interlocutor', 'Up', 'Down', 'Side', 'Other'],
        'Eyebrows': ['Frowning', 'Raising', 'Other'],
        'Eyes': ['X-Open', 'Close-BE', 'Closing-E', 'Close-R', 'Other'],
    },
    'hand': {
        'HandTrajectory': ['Up', 'Down', 'Sideways', 'Complex', 'Other'],
        'Handedness': ['Both-H', 'Single-H', 'Other'],
    },
    'head': {
        'GeneralFace': ['Smile', 'Scowl', 'Other', 'Laugh'],
        'Head': ['Down', 'Down-R', 'Forward', 'Back', 'Side-Tilt', 'Side-Tilt-R', 'Side-Turn', 'Side-Turn-R', 'Waggle',
                 'Other'],
    },
    'mouth': {
        'MouthLips': ['UP-C', 'Down-C', 'Protruded', 'Retracted'],
        'MouthOpenness': ['Open-M', 'Close-M'],
    },
}


def number_of_answers_per_hit(df):
    max_index = 0
    for column_name in df:
        match = RE_ANSWER_WITH_INDEX_COLUMN.match(column_name)
        if match:
            i = int(match.group('index'))
            max_index = max(max_index, i)
    return max_index


def coarse_relabeling(df):
    # Note that this function modifies df content and also returns it.

    answers_per_hit = number_of_answers_per_hit(df)

    # Append a "2" to every new label, because `df.replace` does not support any intersection between
    #   new and old values.
    mapping_to_coarser_labels = {
        'eye': {
            'EyeMovement': {
                'Interlocutor': 'Interlocutor2',
                'Up': 'Up2',
                'Down': 'Down2',
                'Side': 'Side2',
                'Other': 'Other2',
            },
            'Eyebrows': {
                'Frowning': 'Frowning2',
                'Raising': 'Raising2',
                'Other': 'Other2',
            },
            'Eyes': {
                'X-Open': 'X-Open2',
                'Close-BE': 'Other2',
                'Closing-E': 'Other2',
                'Close-R': 'Close-R2',
                'Other': 'Other2',
            },
        },
        'hand': {
            'HandTrajectory': {
                'Up': 'Up2',
                'Down': 'Down2',
                'Sideways': 'Complex2',
                'Complex': 'Complex2',
                'Other': 'Other2',
            },
            'Handedness': {
                'Both-H': 'Movement2',
                'Single-H': 'Movement2',
                'Other': 'Not-Movement2',
            },
        },
        'head': {
            'GeneralFace': {
                'Smile': 'SmileLaugh2',
                'Scowl': 'Scowl2',
                'Other': 'Other2',
                'Laugh': 'SmileLaugh2',
            },
            'Head': {
                'Down': 'Other2',
                'Down-R': 'Nodding2',
                'Forward': 'Other2',
                'Back': 'Other2',
                'Side-Tilt': 'Other2',
                'Side-Tilt-R': 'Other2',
                'Side-Turn': 'Other2',
                'Side-Turn-R': 'Shaking2',
                'Waggle': 'Other2',
                'Other': 'Other2',
            },
        },
        'mouth': {
            'MouthLips': {
                'UP-C': 'UP-C2',
                'Down-C': 'Down-C2',
                'Protruded': 'Protruded2',
                'Retracted': 'Retracted2',
            },
            'MouthOpenness': {
                'Open-M': 'Open-M2',
                'Close-M': 'Close-M2',
            },
        },
    }
    columns_relabeling_dict = {f'Answer.{feature_name}{i}': feature_relabeling
                               for type_dict in mapping_to_coarser_labels.values()
                               for feature_name, feature_relabeling in type_dict.items()
                               for i in range(1, answers_per_hit + 1)}
    return df.replace(columns_relabeling_dict)
