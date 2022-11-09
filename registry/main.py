
import jinja2

import registry

@registry.reg("cuda.conv2d.config")
def conv2d_config():
    """Populates conv2d cutlass configs into 'op_instance' field."""
    print("这个注册函数被运行啦！")

registry.BACKEND_FUNCTIONS["cuda.conv2d.config"]()
