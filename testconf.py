import src.config as conf
import importlib

print(conf.TILE_SIZE)

importlib.reload(conf)
print(conf.TILE_SIZE)
