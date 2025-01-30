import sys
sys.path.append(".")
sys.path.append("..")

from src.dataset import GeoDiversityData

data = GeoDiversityData(
    use_image_cache=True, # Set to false if you want to generate cache from tiffs
    save_image_cache='pickle', # Use 'lzma' to get compressed cache, 'pickle' for uncompressed
    overwrite_image_cache=False # Use if you want to regenerate cache
)
