from abc import ABC, abstractmethod

import numpy as np
import torch
from face_alignment.utils import flip, get_preds_fromhm


class FaceDetector(ABC):

    @abstractmethod
    def run(self, image, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def landmarks_from_batch_no_face_detection(self, images):
        """
        This function is used to get the landmarks from a batch of images without face detection.
        Input:
            images: a batch of images, shape (N, C, H, W), image range [0, 1]
        Returns:
            landmarks: a list of landmarks, each landmark is a numpy array of shape (N, 68, 2), the position is relative
                       ([0, 1])
            landmark_scores: a list of landmark scores, each landmark score is a numpy array of shape (N, 1) or None if
                             no score is available
        """
        raise NotImplementedError()

    def optimal_landmark_detector_im_size(self):
        """
        This function returns the optimal image size for the landmark detector.
        Returns:
            optimal_im_size: int
        """
        raise NotImplementedError()

    def landmark_type(self):
        """
        This function returns the type of landmarks.
        Returns:
            landmark_type: str
        """
        raise NotImplementedError()


class FAN(FaceDetector):

    def __init__(self, device='cuda', threshold=0.5):
        import face_alignment
        self.face_detector = 'sfd'
        self.face_detector_kwargs = {
            "filter_threshold": threshold
        }
        self.flip_input = False
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                                  device=str(device),
                                                  flip_input=self.flip_input,
                                                  face_detector=self.face_detector,
                                                  face_detector_kwargs=self.face_detector_kwargs)

    def run(self, image, with_landmarks=False, detected_faces=None):
        """
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        """
        out = self.model.get_landmarks(image, detected_faces=detected_faces)
        torch.cuda.empty_cache()
        if out is None:
            del out
            if with_landmarks:
                return [], 'kpt68', []
            else:
                return [], 'kpt68'
        else:
            boxes = []
            kpts = []
            for i in range(len(out)):
                kpt = out[i].squeeze()
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                bbox = [left, top, right, bottom]
                boxes += [bbox]
                kpts += [kpt]
            del out  # attempt to prevent memory leaks
            if with_landmarks:
                return boxes, 'kpt68', kpts
            else:
                return boxes, 'kpt68'

    @torch.no_grad()
    def landmarks_from_batch_no_face_detection(self, images):
        out = self.model.face_alignment_net(images).detach()
        if self.flip_input:
            out += flip(self.model.face_alignment_net(flip(images)).detach(), is_label=True)

        out = out.cpu().numpy()
        center = None
        scale = None
        b = out.shape[0]
        pts, pts_img, scores = get_preds_fromhm(out, center, scale)
        pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
        pts, pts_img = pts.view(b, 68, 2) * 4, pts_img.view(b, 68, 2)
        scores = scores
        pts /= images.shape[-1]
        pts = pts.cpu().numpy()
        return pts, scores

    def optimal_landmark_detector_im_size(self):
        # this number is taken from the crop size used in the face_alignment library
        # function in face_alignment.utils.crop : def crop(image, center, scale, resolution=256.0):
        return 256

    def landmark_type(self):
        return 'kpt68'
