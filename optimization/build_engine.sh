$ pip install -q \
    "nvidia-modelopt[onnx]" \
    onnx \
    onnxruntime \
    onnx-graphsurgeon \
    onnxslim \
    onnxscript \
    lief \
    polygraphy



$ python -m modelopt.onnx.autocast \
    --onnx_path ./model.onnx \
    --output_path ./model_trt_fp16.onnx \
    --low_precision_type fp16    




$ trtexec \
  --onnx=./model_trt_fp16.onnx \
  --saveEngine=./autobert-6.engine \
  --minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
  --optShapes=input_ids:4x128,attention_mask:4x128,token_type_ids:4x128 \
  --maxShapes=input_ids:8x256,attention_mask:8x256,token_type_ids:8x256    


  