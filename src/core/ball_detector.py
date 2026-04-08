from src.core.tracknet import BallTrackerNet
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm


class BallDetector:
    def __init__(self, path_model=None, device="cuda"):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        self._using_cpu_fallback = False
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 360

    def infer_model(self, frames, verbose=False):
        """Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
        """
        ball_track = [(None, None)] * 2
        prev_pred = [None, None]
        for num in tqdm(range(2, len(frames)), disable=not verbose):
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num - 1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num - 2], (self.width, self.height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            inp_tensor = torch.from_numpy(inp).float().to(self.device)
            try:
                with torch.inference_mode():
                    if self.device == "cuda" and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            out = self.model(inp_tensor)
                    else:
                        out = self.model(inp_tensor)
            except torch.cuda.OutOfMemoryError:
                if self.device != "cuda":
                    raise

                # Free fragmented CUDA cache, then switch to CPU for stability.
                torch.cuda.empty_cache()
                if not self._using_cpu_fallback:
                    print("[WARNING]: CUDA OOM in ball detector, switching to CPU inference.")
                    self.model = self.model.to("cpu")
                    self.model.eval()
                    self.device = "cpu"
                    self._using_cpu_fallback = True

                del inp_tensor
                inp_tensor = torch.from_numpy(inp).float().to(self.device)
                with torch.inference_mode():
                    out = self.model(inp_tensor)

            output = out.argmax(dim=1).cpu().numpy()
            
            # Clean up tensors immediately
            del inp_tensor, out
            
            x_pred, y_pred = self.postprocess(output, prev_pred)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
            
            # Clean up intermediate arrays (after postprocess is done)
            del img, img_prev, img_preprev, imgs, inp, output
        return ball_track

    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            scale: scale for conversion to original shape (720,1280)
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=2,
            minRadius=2,
            maxRadius=7,
        )
        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0] * scale
                    y_temp = circles[0][i][1] * scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break
            else:
                x = circles[0][0][0] * scale
                y = circles[0][0][1] * scale
        return x, y
