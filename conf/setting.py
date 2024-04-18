from conf.packages import *
from conf.Tools import Tools
from utils import *
tools = Tools()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# log_file = open("res/log/log.txt", "a", encoding="utf-8")

console = Console(record=False, soft_wrap=True)

