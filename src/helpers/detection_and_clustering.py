import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
import dlib
import face_recognition
import helpers.general as gen


def detect_faces(cap, vid_name, data_path, min_size=50):
    vid_faces = []
    frame_count = 0
    rotate = False
    ret, frame = cap.read()

    # checks if we need to rotate the video
    rotate = gen.rotate_value(data_path + 'video/' + vid_name + '.mp4')

    while ret:
        # rotates if necissary
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # detects and saves faces
        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model='cnn')
        for (top, right, bottom, left) in face_locations:
            # only takes faces that are past a minimum size
            if (bottom - top) >= min_size or (right - left) >= min_size:
                encodings = face_recognition.face_encodings(face_image=frame, known_face_locations=[(top, right, bottom, left)], model='large')
                vid_faces.append([vid_name, frame_count, (top, right, bottom, left)] + list(encodings[0]))

        ret, frame = cap.read()
        frame_count += 1

    return vid_faces


def cluster_faces(vid_faces, clustering_threshold=0.5):
    # converts the encodings to dlib.vectors and does some clustering
    encodings = [v[-128:] for v in vid_faces]
    encodings = [dlib.vector(e) for e in encodings]
    cluster = dlib.chinese_whispers_clustering(encodings, clustering_threshold)
    print(f'Number of clusters: {len(set(cluster))}')

    # makes a data frame and adds the cluster ids
    df_faces = pd.DataFrame([v[:3] for v in vid_faces], columns=['vid_id', 'frame', 'position'])
    df_faces['cluster'] = cluster
    df_encodings = pd.DataFrame([v[-128:] for v in vid_faces], columns=['encoding_' + str(i) for i in range(128)])
    df_faces = df_faces.join(df_encodings)

    return df_faces


def get_cluster_center(df):
    df_return = None
    for cluster in df['cluster'].unique():
        # gets just the encodings
        df_cluster = df.loc[df['cluster'] == cluster, :].reset_index(drop=True)
        encodings = df_cluster.iloc[:, -128:]

        # calcuates the average of the cluster
        encodings = encodings.to_numpy()
        m = np.mean(encodings, 0)

        # calculate the cos distance of each face
        cos_distance = [np.dot(e, m) / (np.linalg.norm(e) * np.linalg.norm(m)) for e in encodings]

        # finds the max and keeps that one, cosin distance is [-1, 1] and 1 is perfect match
        df_cluster = df_cluster.iloc[[cos_distance.index(max(cos_distance))], :]

        # appeds to the results dataframe
        if df_return is None:
            df_return = df_cluster
        else:
            df_return = df_return.append(df_cluster)

    return df_return


def cluster_faces(df, clustering_threshold=0.5):
    # gets the encodings and clusters on them
    encodings = gen.get_encodings(df)
    encodings = [dlib.vector(e) for e in encodings]
    clusters = dlib.chinese_whispers_clustering(encodings, clustering_threshold)
    print(f'Number of clusters: {len(set(clusters))}')
    df['cluster'] = clusters

    # gets the size of each cluster
    df_clusters_size = df.loc[:, ['vid_id', 'cluster']]
    df_clusters_size = df_clusters_size.groupby('cluster')['vid_id'].count().reset_index()
    df_clusters_size.columns = ['cluster', 'size']
    df = pd.merge(df, df_clusters_size, on='cluster', how='left')
    df = df.sort_values('size', ascending=False)
    print(f"Number of clusters with n > 1: {len(df.loc[df['size'] > 1, 'cluster'].unique())}")

    # reorders the columns and returns the dataframe
    df = gen.order_df(df)
    return df


def find_closest_cluster(df, data_path, img_path, sim_cutoff=0.5):
    print(f'{img_path}================================')
    # get the image encodings
    img = cv2.imread(img_path)
    # first try to find a face in the image
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model='cnn')
    # find the largest face or uses the entire image
    if len(face_locations) > 0:
        print('Found face')
        max_size = 0
        max_index = 0
        for i in range(len(face_locations)):
            top, right, bottom, left = face_locations[i]
            size = (bottom - top) * (right - left)
            if size > max_size:
                max_index = i

        # saves the encodings
        top, right, bottom, left = face_locations[max_index]
        face_location = [(top, right, bottom, left)]
    else:
        face_location = [(0, img.shape[1], img.shape[0], 0)]

    img_encodings = face_recognition.face_encodings(face_image=img, known_face_locations=face_location, model='large')
    img_encodings = img_encodings[0]

    # crops the image to just the face
    img = img[face_location[0][0]:face_location[0][2], face_location[0][3]:face_location[0][1]]

    df = df.reset_index(drop=True)
    encodings = gen.get_encodings(df)
    cos_similarity = [np.dot(e, img_encodings) / (np.linalg.norm(e) * np.linalg.norm(img_encodings)) for e in encodings]

    # gets the index of the most similar face
    max_similarity = max(cos_similarity)
    if max_similarity >= sim_cutoff:
        print('Submitted image')
        img = cv2.resize(img, (200, 200))
        cv2_imshow(img)

        # prints out the closest match
        max_index = cos_similarity.index(max_similarity)
        matching_cluster = df.loc[max_index, 'cluster']
        print(f'Closest match score: {max_similarity}')
        img = cv2.imread(f'{data_path}/faces/{df.loc[max_index, "id"]}.jpg')
        img = cv2.resize(img, (200, 200))
        cv2_imshow(img)

        # prints out the closest cluster
        print(f'Cluster match: {matching_cluster}')
        gen.show_cluster(df, matching_cluster, data_path)

        # prints out the video names
        print('Videos used in this cluster:')
        print(df.loc[df['cluster'] == matching_cluster, 'vid_id'].to_list())
