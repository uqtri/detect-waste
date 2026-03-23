import albumentations as A

def get_transform():
    transforms = A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomSizedBBoxSafeCrop(700, 700, erosion_rate=0.0, p=0.5),
        A.Blur(blur_limit=7, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.5),
        # A.Downscale(scale_range=(0.5, 0.9), p=0.5),
        # A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=4, p=1.0),
        # A.RandomFog(fog_coef_range=(0.3, 1.0), alpha_coef=0.08, p=0.2),
        # A.RandomRain(p=0.2),
        # A.RandomSnow(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes'])
    )
    return transforms
    