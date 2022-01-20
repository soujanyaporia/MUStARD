#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
pushd "${SCRIPT_DIR}/.." > /dev/null

videos_folder_path=data/videos/utterances_final
audios_folder_path=data/audios/utterances_final
ext=mp4

mkdir -p "${audios_folder_path}"

for video_file_path in "${videos_folder_path}"/*."${ext}"; do
    slash_and_video_file_name="${video_file_path:${#videos_folder_path}}"
    slash_and_video_file_name_without_extension="${slash_and_video_file_name%.${ext}}"
    audio_file_path="${audios_folder_path}${slash_and_video_file_name_without_extension}.aac"
    ffmpeg -i "${video_file_path}" -vn -acodec copy "${audio_file_path}"
done

popd > /dev/null
