import os
import cv2
import random
from collections import defaultdict
import numpy as np

class Augmentation(object):
    def __init__(self, params):
        self.params = params
        self.img = None
        self.resize = None
        self.flip = None
        self.clahe = None
        self.equilize = None
        self.scale = None
        self.rotate = None
        self.augmented_images = defaultdict(list)
        self.n_augment = 0
        self.fc = 0
        self.update_params()
        print ('Total files after augmentation = %d' % self.n_augment)

    def update_params(self):
        if 'resize' in self.params:
            self.resize = True
            self.n_augment += 1
        if 'flip' in self.params:
            self.flip = True
            self.n_augment += 1
        if 'clahe' in self.params:
            self.clahe = True
            self.n_augment += len(self.params['clahe'][0])
        if 'equilize' in self.params:
            self.equilize = True
            self.n_augment += 1
        if 'equilize' in self.params:
            self.scale = self.params['scale']
            self.n_augment += len(self.params['scale'][0])

    def BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def img_load(self, gray=1):
        if os.path.exists(self.filename):
            self.img = cv2.imread(self.filename, int(gray!=1))

    def img_resize(self):
        if self.resize is not None:
            self.img = cv2.resize(self.img, self.params['resize'])
            self.augmented_images['resized'].append([self.BGR(self.img), self.labels])

    def img_flip(self):
        if self.flip is not None:
            for fp in self.params['flip']:
                flp = cv2.flip(self.img, fp)
                h, w = flp.shape[:2]
                if self.labels is not None:
                    labels = []
                    for lab in self.labels:
                        labels.append((1.0-lab[0], lab[1]))
                else:
                    labels = self.labels
                self.augmented_images['flipped'].append([self.BGR(flp), labels])

    def img_clahe(self):
        if self.clahe is not None:
            randomize = self.params['clahe'][1]
            for clip in self.params['clahe'][0]:
                if randomize:
                    clip1 = clip
                    if self.params['clahe'][1]:
                        clip1 = random.uniform(clip-1, clip)
                    clahe = cv2.createCLAHE(clipLimit=clip1, tileGridSize=(8, 8))
                    cl1 = clahe.apply(self.img)
                    self.augmented_images['clahe'].append([self.BGR(cl1), self.labels])

    def img_equilize(self):
        if self.equilize is not None:
            equ = cv2.equalizeHist(self.img)
            self.augmented_images['equilized'].append([self.BGR(equ), self.labels])

    def img_scale(self):
        if self.scale is not None:
            for s in self.params['scale'][0]:
                s1 = s
                if s1 < 1.0:
                    if self.params['scale'][1]:
                        s1 = random.uniform(s, 1.0)
                    sc = cv2.resize(self.img, (int(self.img.shape[1]*s1), int(self.img.shape[0]*s1)))
                    sc1 = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
                    x = int((self.img.shape[1]-sc.shape[1])/2.0)
                    y = int((self.img.shape[0]-sc.shape[0])/2.0)
                    sc1[y:y+sc.shape[0], x:x+sc.shape[1]] = sc
                    if self.labels is not None:
                        labels, wf, hf = [], sc.shape[1]/sc1.shape[1], sc.shape[0]/sc1.shape[0]
                        for lab in self.labels:
                            labels.append((lab[0]*wf, lab[1]*hf))
                    else:
                        labels = self.labels
                else:
                    if self.params['scale'][1]:
                        s1 = random.uniform(1.0, s)
                    sc = cv2.resize(self.img, (int(self.img.shape[1]*s1), int(self.img.shape[0]*s1)))
                    x = int((sc.shape[1]-self.img.shape[1])/2.0)
                    y = int((sc.shape[0]-self.img.shape[0])/2.0)
                    sc1 = sc[y:y+self.img.shape[0], x:x+self.img.shape[0]]
                    if self.labels is not None:
                        labels, wf, hf = [], sc1.shape[1]/sc.shape[1], sc1.shape[0]/sc.shape[0]
                        for lab in self.labels:
                            labels.append((lab[0]*wf, lab[1]*hf))
                    else:
                        labels = self.labels
                self.augmented_images['scaled'].append([self.BGR(sc1), labels])    

    def img_rotate(self):
        for rot in self.params['rotate'][0]:
            degree = rot
            if self.params['rotate'][1]:
                if degree < 0:
                    degree = random.uniform(degree, 0.0)
                else:
                    degree = random.uniform(0.0, degree)
            h, w = self.img.shape[:2]
            rmat = cv2.getRotationMatrix2D((w/2, h/2), degree, 1)
            rimg = cv2.warpAffine(self.img, rmat, (w, h))
            self.augmented_images['rotated'].append([self.BGR(rimg), self.labels])    

    def perform(self, filename, labels=None):
        self.filename = filename
        self.labels = labels
        
        self.img_load()
        self.img_resize()
        self.img_flip()
        self.img_clahe()
        self.img_scale()
        self.img_rotate()

        # return all aumented images with respective labels
        return self.augmented_images

    def save_csv(self, path=None, csv=None):
        if path is None:
            path = os.path.dirname(self.filename)

        if csv is None:
            csv = 'output.csv'

        out = open(csv, 'a')

        for values in self.augmented_images:
            for i, data in enumerate(augmented[values]):
                img = data[0]
                fname = os.path.basename(self.filename)[:-4]
                fname = '%s_%s%d.jpg' % (path, values, i)
                fpath = os.path.join(path, fname)
                cv2.imwrite(fpath, img)
                out.write('%d,%s,%s\n' % (self.fc, fname, ','.join(self.labels)))
                self.fc += 1
        out.close()

    def show(self):
        for values in self.augmented_images:
            for i, data in enumerate(self.augmented_images[values]):
                img = data[0]
                h, w = img.shape[:2]
                if data[1] is not None:
                    for lab in data[1]:
                        cv2.circle(img, (int(lab[0]*w), int(lab[1]*h)), 3, (0, 255, 0), -1)
                cv2.imshow('image-%s-%d' % (values, i), img)
                cv2.waitKey(0)


def main():
    aug = Augmentation({'resize': (128, 128), 'clahe': [[1, 2, 3], True], 'flip': [1], 
                        'equilize': [], 'scale': [(0.8, 1.2), True], 'rotate': [(-30, 30), True]})
    files_list = ['/home/danish/Downloads/lena.png']
    labels_list = [[(120/256.0, 120/256.0), (180/256.0, 120/256.0), (180/256.0, 180/256.0), (120/256.0, 180/256.0)]]
    for f, files in enumerate(files_list):
        # augmented = aug.perform(files, labels_list[f])
        augmented = aug.perform(files)
        aug.show()
if __name__=='__main__':
    main()
