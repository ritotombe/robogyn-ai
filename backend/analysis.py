"""Research-only sampled whole-frame motion descriptors; NOT skill scores."""
from pathlib import Path
import cv2
import numpy as np

class VideoError(ValueError):
    pass

def inspect(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError('Video cannot be decoded')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or fps > 240 or frames < 2 or width < 32 or height < 32:
            raise VideoError('Invalid or unsupported video metadata')
        duration = frames / fps
        if duration > 4 * 3600:
            raise VideoError('Video longer than 4 hours is not supported')
        return {'fps': round(fps, 3), 'frame_count': frames, 'width': width, 'height': height, 'duration_s': round(duration, 3)}
    finally:
        cap.release()

def analyze(path: Path, start_s: float, end_s: float, sample_fps: float = 2.0, crop_top_fraction: float = 0.1) -> dict:
    meta = inspect(path)
    if not (0 <= start_s < end_s <= meta['duration_s'] + 0.001):
        raise VideoError('Invalid start/end; must fall within the video')
    if end_s - start_s < 2 or end_s - start_s > 300:
        raise VideoError('Select a segment between 2 and 300 seconds')
    if sample_fps not in (0.5, 1.0, 2.0, 4.0):
        raise VideoError('sample_fps must be 0.5, 1, 2, or 4')
    if not 0 <= crop_top_fraction <= 0.4:
        raise VideoError('crop_top_fraction must be between 0 and 0.4')
    cap = cv2.VideoCapture(str(path))
    observations = []
    prev = None
    try:
        # Sampling by target timestamp avoids misleading claims of native-frame tracking.
        times = np.arange(start_s, end_s, 1.0 / sample_fps)
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            h = frame.shape[0]
            frame = frame[int(h * crop_top_fraction):]
            if frame.size == 0:
                continue
            scale = min(1.0, 480 / max(frame.shape[:2]))
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if prev is not None and prev.shape == gray.shape:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = cv2.magnitude(flow[..., 0], flow[..., 1])
                observations.append({'time_s': round(float(t), 3),
                                     'mean_flow_px_per_sample': round(float(np.mean(mag)), 4),
                                     'p90_flow_px_per_sample': round(float(np.percentile(mag, 90)), 4)})
            prev = gray
    finally:
        cap.release()
    if not observations:
        raise VideoError('Insufficient readable frames; use a longer segment or different encoding')
    return {'task': 'robotic_suturing', 'experience_level': 'beginner',
            'segment': {'start_s': start_s, 'end_s': end_s, 'duration_s': round(end_s - start_s, 3)},
            'sampling_fps': sample_fps, 'crop_top_fraction': crop_top_fraction,
            'summary': {'mean_flow_px_per_sample': round(float(np.mean([o['mean_flow_px_per_sample'] for o in observations])), 4),
                        'max_p90_flow_px_per_sample': round(float(max(o['p90_flow_px_per_sample'] for o in observations)), 4),
                        'sample_count': len(observations)},
            'observations': observations,
            'interpretation': 'Exploratory whole-frame optical flow only; camera and tissue movement confound measurements. Not a GEARS score or proficiency assessment.'}
