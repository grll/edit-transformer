from typing import Union

from torch import LongTensor
from torch.cuda import LongTensor as CudaLongTensor


T_LongTensor = Union[LongTensor, CudaLongTensor]