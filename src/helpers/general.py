import cv2
import numpy as np
import ffmpeg
from tqdm import tnrange
from google.colab.patches import cv2_imshow


def rotate_value(vid_path):
    # checks if we need to rotate the video
    rotate = False
    meta_data = ffmpeg.probe(vid_path)['streams'][0]
    try:
        if meta_data['tags']['rotate'] == '90':
            rotate = True
    except KeyError:
        pass

    return rotate


def get_video_summary(vid_list, data_path):
    total_sec = 0
    total_frames = 0
    no_fps_count = 0
    for i in tnrange(len(vid_list)):
        cap = cv2.VideoCapture(data_path + 'video/' + vid_list[i])
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames += frame_count
            total_sec += (frame_count / fps)
        else:
            print(f'Not found: {vid_list[i]}')
            no_fps_count += 1

    print(f'Number of videos: {len(vid_list)}')
    print(f'Number of frames: {total_frames}')
    print(f'Total minutes: {(total_sec / 60):.2f}')
    print(f'Total Hours: {(total_sec / 3600):.2f}')


def get_cluster_faces(df, data_path, resize=200):
    img_cluster = None
    for i, row in df.iterrows():
        rotate = False
        cap = cv2.VideoCapture(data_path + 'video/' + row['video'] + '.mp4')
        ret, frame = cap.read()

        # checks if we need to rotate the video
        rotate = rotate_value(data_path + 'video/' + row['video'] + '.mp4')

        # finds the face in the video
        frame_count = 0
        while ret:
            if frame_count == row['frame']:
                # rotates if necissary
                if rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                pos = row['position']
                if isinstance(pos, tuple) is False:
                    pos = eval(pos)
                img_face = frame[pos[0]:pos[2], pos[3]:pos[1]]
                break

            frame_count += 1
            ret, frame = cap.read()

        # resizes it and appends it to the other faces
        img_face = cv2.resize(img_face, (200, 200))
        if img_cluster is None:
            img_cluster = img_face
        else:
            img_cluster = np.hstack((img_cluster, img_face))

    return img_cluster


def show_clusters(df, data_path, img_sample=5, cluster_sample=5):
    clusters = df['cluster'].unique()
    # random.shuffle(clusters)
    clusters = clusters[:cluster_sample]
    df_images = None
    img_all_clusters = None
    for cluster in clusters:
        # gets a single cluster and shuffles the faces
        df_temp = df.loc[df['cluster'] == cluster, :].reset_index(drop=True)
        if img_sample <= df_temp.shape[0]:
            df_temp = df_temp.sample(img_sample)
        else:
            df_temp = df_temp.sample(df_temp.shape[0])
        df_temp = df_temp.loc[:, ['video', 'frame', 'position', 'cluster']]
        img_cluster = get_cluster_faces(df_temp, data_path)

        # shows the faces of the cluster
        cv2_imshow(img_cluster)


def show_face_location(df, data_path):
    for i, row in df.iterrows():
        file_name = data_path + 'video/' + row['video'] + '.mp4'
        cap = cv2.VideoCapture(file_name)
        print(cap.isOpened())
        ret, frame = cap.read()

        # checks if we need to rotate the video
        rotate = rotate_value(data_path + 'video/' + row['video'] + '.mp4')

        # loops through to the spot of the face
        frame_count = 0
        while ret:
            if frame_count == row['frame']:
                # rotates if necissary
                if rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                pos = row['position']
                frame = cv2.rectangle(frame, (pos[3], pos[0]), (pos[1], pos[2]), (255, 0, 0), 5)
                cv2_imshow(frame)
                break

            frame_count += 1
            ret, frame = cap.read()


def get_encodings(df):
    encoding_columns = ['encoding_' + str(i) for i in range(128)]
    encodings = df.loc[:, encoding_columns]
    encodings = encodings.to_numpy()
    return encodings


def order_df(df):
    encoding_columns = [f'encoding_{i}' for i in range(128)]
    first_columns = [c for c in list(df.columns) if c not in encoding_columns]
    df = df.loc[:, first_columns + encoding_columns]
    return df


def get_similarity(df):
    clusters = df['cluster'].unique()
    df['similarity'] = -1

    for cluster in clusters:
        df_temp = df.loc[df['cluster'] == cluster, :]
        if df_temp.shape[0] > 1:
            encodings = get_encodings(df_temp)

            # calculate the average cos similarity of each face to the average value
            m = np.mean(encodings, 0)
            cos_similarity = [np.dot(e, m) / (np.linalg.norm(e) * np.linalg.norm(m)) for e in encodings]
            cos_similarity = np.mean(cos_similarity)
            df.loc[df['cluster'] == cluster, 'similarity'] = cos_similarity

    df = df.sort_values(['similarity', 'cluster'], ascending=False)
    df = order_df(df)
    return df


def show_cluster(df, cluster_id, data_path, img_sample=5, shuffle=True):
    df = df.loc[df['cluster'] == cluster_id, :]

    faces = None
    if shuffle:
        df = df.sample(df.shape[0]).reset_index(drop=True)
        df = df.iloc[:5, :]

    # gets the images and appends them
    for i, row in df.iterrows():
        img = cv2.imread(f'{data_path}faces/{row["id"]}.jpg')
        img = cv2.resize(img, (200, 200))

        # stacks the faces next to each other
        if faces is None:
            faces = img
        else:
            faces = np.hstack((faces, img))

    # shows the faces of the cluster
    cv2_imshow(faces)
