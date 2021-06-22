import fnmatch
import imageio
import os
import rawpy
from shutil import copyfile
import time_util
from pathlib import Path

""" Convert DNG to JPG with some post processing to make up for demosaicing 

1. Open Image given a Path
2. Post Process Raw to RGB
3. Save as JPG file type given output name

  Typical usage example: 

  Example convert(image_path, outputname)
"""


def convert(file_path, out_path):
	out_path = Path(out_path)
	if Path(file_path).suffix in [".dng", ".DNG"]:
		with rawpy.imread(file_path) as raw:
			# https://letmaik.github.io/rawpy/api/rawpy.Params.html
			rgb = raw.postprocess(demosaic_algorithm=0, use_camera_wb=True, four_color_rgb=False,
                                  no_auto_bright=True, exp_shift=2.5, user_black=350, exp_preserve_highlights=1.0)
		print("file_path: ",file_path)
		os.remove(file_path)
		out_path = str(file_path)[:-4]
		print(out_path + '.jpg')
		imageio.imwrite(out_path + '.jpg', rgb)
		return str(out_path +'.jpg')
	# else:
	# 	print(str(out_path / file_path.name))
	# 	return str(out_path / file_path.name)
