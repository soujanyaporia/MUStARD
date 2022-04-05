# Visual Feature Extraction

Follow these steps to extract visual features from the MUStARD dataset videos.

1. Download the videos from Google Drive to `data/videos`, placing the files there without subdirectories.
2. Move to this directory:

    ```bash
    cd visual
    ```

3. Run `save_frames.sh` to extract the frames from the video files:

    ```bash
    ./save_frames.sh
    ```

4. Create the directories, `data/features/`, `data/features/utterances_final/` and `data/features/context_final/`.

5. To extract the features and save them into large H5 files:

    ```bash
    ./extract_features.py resnet
    ``` 

    * If you extract C3D features, first download
    [the Sports1M-pretrained C3D weights](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)
    into `data/features/c3d.pickle`.
    * If you extract I3D features, first download
    [the ImageNet-and-Kinetics-400-pretrained I3D weights](https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt)
    into `data/features/i3d.pt`.
