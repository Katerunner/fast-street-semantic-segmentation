import pandas as pd
import numpy as np
import tqdm
import cv2


def load_dataset_into_memory(guide_path='dataset/guide.csv', input_shape=(128, 128)):
    guide = pd.read_csv(guide_path)
    guide['image'] = [path.replace('WildDash2', 'dataset') for path in guide['image']]
    guide['mask'] = [path.replace('WildDash2', 'dataset') for path in guide['mask']]

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for i in tqdm.trange(len(guide), position=0, leave=True):
        try:
            sample_image_data = guide.iloc[i]
            image = cv2.cvtColor(cv2.resize(cv2.imread(sample_image_data['image']), input_shape), cv2.COLOR_BGR2RGB)
            label = cv2.resize(cv2.imread(sample_image_data['mask']), input_shape)[:, :, 0]
            if sample_image_data['type'] == 'train':
                X_train.append(image)
                y_train.append(label)
            else:
                X_test.append(image)
                y_test.append(label)
        except Exception as a:
            print(a)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test
