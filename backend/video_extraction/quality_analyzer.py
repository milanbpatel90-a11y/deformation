"""Resolution-aware quality metrics for eyewear reference frames."""
import cv2, numpy as np

def analyze_quality(image):
    if image is None or image.size == 0:
        return {"sharpness": 0.0, "brightness": 0.0, "contrast": 0.0, "quality_score": 0.0, "reasons": ["invalid_frame"]}
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); h,w=gray.shape
    scale=min(1.0, 960.0/max(h,w)); small=cv2.resize(gray,None,fx=scale,fy=scale) if scale<1 else gray
    sharp=float(cv2.Laplacian(small,cv2.CV_64F).var()); brightness=float(gray.mean()); contrast=float(gray.std())
    # Normalize sharpness against the sampled image resolution; this is a ranking score, not a universal pass/fail truth.
    sharp_score=float(np.clip(np.log1p(sharp)/np.log1p(180.0),0,1))
    exposure_score=float(np.clip(1-abs(brightness-128)/128,0,1)); contrast_score=float(np.clip(contrast/64,0,1))
    reasons=[]
    if sharp_score<0.28: reasons.append("blurry")
    if brightness<35: reasons.append("underexposed")
    if brightness>225: reasons.append("overexposed")
    if contrast<14: reasons.append("low_contrast")
    score=0.55*sharp_score+0.25*exposure_score+0.20*contrast_score
    return {"sharpness":round(sharp,3),"sharpness_score":round(sharp_score,4),"brightness":round(brightness,2),"contrast":round(contrast,2),"quality_score":round(float(score),4),"reasons":reasons}
