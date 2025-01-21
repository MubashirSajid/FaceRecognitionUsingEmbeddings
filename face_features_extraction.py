from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import torch
from facenet_pytorch.models.utils.detect_face import extract_face
from PIL import Image


class Whitening(object):
    def __call__(self, img):
        mean = img.mean()
        std = img.std()
        std_adj = std.clamp(min=1.0 / (float(img.numel()) ** 0.5))
        y = (img - mean) / std_adj
        return y



class FacialFeaturesExtractor:
    def __init__(self):
        self.aligner = MTCNN(keep_all=True, thresholds=[0.8, 0.9, 0.95])
        self.facenet_preprocess = transforms.Compose([Whitening()])
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval()

    def extract_features(self, img):
        bbs, _ = self.aligner.detect(img)
        if bbs is None:
            return None, None
        
        faces = torch.stack([extract_face(img,bb) for bb in bbs])
        embeddings = self.facenet(self.facenet_preprocess(faces )).detach().numpy()

        return bbs, embeddings

    def __call__(self, img):
        return self.extract_features(img=img)
    

exif_orientation_tag = 0x0112
exif_transpose_sequences = [
    [],
    [],
    [Image.FLIP_LEFT_RIGHT],
    [Image.ROTATE_180],
    [Image.FLIP_TOP_BOTTOM],
    [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],
    [Image.ROTATE_270],
    [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],
    [Image.ROTATE_90]
]
class ExifOrientationNormalize(object):
    def __call__(self, img):
        if 'parsed_exif' in img.info and exif_orientation_tag in img.info['parsed_exif']:
            orientation = img.info['parsed_exif'][exif_orientation_tag]
            transposes = exif_transpose_sequences[orientation]
            for trans in transposes:
                img = img.transpose(trans)
            
            return img