import cv2

from config import Config
from log import logger

from mesh import Mesh
from camo import Camo
from render import Renderer
from utils import convert_to_numpy

config = Config(logger, './config/base.yaml').item()
logger.set_config(config)

rd = Renderer(config)
ms = Mesh(config)
camo = Camo(config, ms.shape())
camo.load_mask()
ms.set_camo(camo)
mesh = ms.item()
rd.set_camera_position(8.0, 47.40421576000235, 120.72393414768057)
image = rd.render(mesh)
image = convert_to_numpy(image)
cv2.imwrite("output/test.png", image)

ms.make_texture_map_from_atlas()
