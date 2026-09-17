from typing import Any
import numpy as np
import numpy.typing as npt
import onnxruntime as rt
import os 
import yaml


rt.preload_dlls()

PREFERRED_CUDA_DEVICE = 0

_ORT_TO_NP_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}

class onnx_session:
    def __init__(self, model_path:str):
        self.session  = rt.InferenceSession(model_path,providers = ['CUDAExecutionProvider','CPUExecutionProvider'])
        self._i_list = self.session.get_inputs()
        self._o_list = self.session.get_outputs()
        assert not _is_dynamic(self._i_list,self._o_list), "Dynamic model shape detected - incompatible"
        assert rt.get_device() == "GPU","Onnx provider fell back to CPU, CUDA unavailable"
        self._io = self.session.io_binding()

        #initialise input and output GPU buffers and bind them to the model IO
        self._i = [rt.OrtValue.ortvalue_from_numpy(
            np.zeros(i.shape,_ORT_TO_NP_DTYPE[i.type]),
            device_type="cuda",
            device_id=PREFERRED_CUDA_DEVICE
            )
            for i in self._i_list]
        for (j,_i) in enumerate(self._i):self._io.bind_ortvalue_input(self._i_list[j].name,_i)

        self._o = [
            rt.OrtValue.ortvalue_from_shape_and_type(
                o.shape, 
                _ORT_TO_NP_DTYPE[o.type],
                device_type="cuda",
                device_id=PREFERRED_CUDA_DEVICE
                )
            for o in self._o_list
        ]
        for (j,_o) in enumerate(self._o):self._io.bind_ortvalue_output(self._o_list[j].name,_o)

        
        
    def __call__(self, i:list[npt.NDArray] | npt.NDArray)->list[npt.NDArray]:
        #do inplace update of self._i (pre-allocated)
        i = [i] if isinstance(i,np.ndarray) else i
        [_i.update_inplace(i[j]) for (j,_i) in enumerate(self._i)]
        self.session.run_with_iobinding(self._io)
        return [_o.numpy() for _o in self._o]

    
class inference_pipeline:
    def __init__(self,config:str):
        with open(config,'r') as file:
            _cfg = yaml.safe_load(file)

        

def _is_dynamic(i,o) -> bool:
    shapes = [_i.shape for _i in i] + [_i.shape for _i in o] 
    shapes = [item for sublist in shapes for item in sublist]
    return any(not isinstance(d, int) for d in shapes)


def main():
    onnx_session("checkpoints/detector_882k.onnx")
if __name__ == "__main__":
    main()