# py-mlmodelscope 

MLModelScope 

# [Documentation](https://docs.mlmodelscope.org/)

The current landscape of Machine Learning (ML) and Deep Learning (DL) is rife with non-uniform models, frameworks, and system stacks but lacks standard tools to evaluate and profile models or systems.
Due to the absence of such tools, the current practice for evaluating and comparing the benefits of proposed AI innovations (be it hardware or software) on end-to-end AI pipelines is both arduous and error prone --- stifling the adoption of the innovations.

MLModelScope is a hardware/software agnostic, extensible and customizable platform for evaluating and profiling ML models across datasets/frameworks/hardware, and within AI application pipelines.
MLModelScope lowers the cost and effort for performing model evaluation and profiling, making it easier for others to reproduce, evaluate, and analyze acurracy or performance claims of models and systems.

It is designed to aid in:

1. reproducing and comparing with published models, and designing models with performance and deployment in mind,
2. understanding the model performance (within realworld AI workflows) and its interaction with all levels of the hardware/software stack
3. discovering models, frameworks and hardware that are applicable to users' datasets.

To achieve this, MLModelScope:

  - Provides a consistent evaluation, aggregation, and reporting system by defining
  - techniques to specify and provision workflows with HW/SW stacks
  - abstractions for evaluation and profiling using different frameworks
  - data consumption for evaluation outputs
  - Enables profiling of experiments throughout the entire pipeline and at different abstraction levels (application, model, framework, layer, library and hardware, as shown on the right)
  - Is framework and hardware agnostic - with current support for TensorFlow, MXNet, TensorRT, Caffe, Caffe2, CNTK running on X86, PowerPC, and ARM CPU with GPU and FPGA
  - Is extensible and customizable - allowing users to extend MLModelScope by adding models, frameworks, or library and system profilers.
  - Can run experiments on separate machines, and behind firewall (does not exposing model weights or machine specification)
  - Allows parallel evaluation (multiple instantiations of the same experiment set-up across systems)
  - Specifies model and framework resources as asset files which can be added easily, even at runtime


MLModelScope can be used as an application with a command line, API or web interface, or can be compiled into a standalone library. We also provide an online hub of continuously updated assets, evaluation results, and access to hardware resources — allowing users to discover and evaluate models without installing or configuring systems.

# Bare Minimum Installation

## Requirements 

``` 
python>=3.8 
cuda>=11.7.1 
numpy>=1.23.5 
tensorflow.=2.12.0 or latest
tensorflow>=1.7.0
``` 
### The CUDA Library

Please refer to Nvidia CUDA library installation on this. Find the localation of your local CUDA installation, which is typically at `/usr/local/cuda/`, and setup the path to the `libcublas.so` library. 

### The CUPTI Library

Please refer to Nvidia CUPTI library installation on this. Find the localation of your local CUPTI installation, which is typically at `/usr/local/cuda/extras/CUPTI`, and setup the path to the `libcupti.so` library. 

Also, please install Pre-requsite Dynamic Library. 

**On Linux**

```bash
cd pycupti/csrc 
export PATH="/usr/local/cuda/bin:$PATH" 
nvcc -O3 --shared -Xcompiler -fPIC utils.cpp -o libutils.so -lcuda -lcudart -lcupti -lnvperf_host -lnvperf_target -I /usr/local/cuda/extras/CUPTI/include -L /usr/local/cuda/extras/CUPTI/lib64 
```

**On Windows**

```console
cd pycupti/csrc 
nvcc -O3 --shared utils.cpp -o utils.dll -I"%CUDA_PATH%/include" -I"%CUDA_PATH%/extras/CUPTI/include" -L"%CUDA_PATH%"/extras/CUPTI/lib64 -L"%CUDA_PATH%"/lib/x64 -lcuda -lcudart -lcupti -lnvperf_host -lnvperf_target -Xcompiler "/EHsc /GL /Gy /O2 /Zc:inline /fp:precise /D "_WINDLL" /Zc:forScope /Oi /MD" && del utils.lib utils.exp 
```

After running above commands, please check whether  `libutils.so` on Linux or `utils.dll` on Windows is in `pycupti/csrc` directory. 
### The Tensorflow Library 

The Tensorflow library is required for our tensorflow-agent. 

You can install Tensoflow by referencing [Tensorflow](https://www.tensorflow.org/install). 

## Test Installation

With the configuration and the above bare minimumn installation, you should be ready to test the installation and see how things works. 

To run an inference using the default DNN model `ResNet_v1_101` with default test input images. 

```bash
python run_image_classification.py --agent tensorflow --model_name ResNet_v1_101
``` 

# External Service Installation to Enable Tracing and Profiling

We now discuss how to install a few external services that make the agent fully useful in terms of collecting tracing and profiling data.

## External Services

MLModelScope relies on a few external services. These services provide tracing functionality.

### Install OpenTelemetry library 

```bash 
pip install opentelemetry-api 
pip install opentelemetry-sdk 
pip install opentelemetry-exporter-otlp-proto-grpc 
pip install grpcio 
``` 

### Starting Trace Server

This service is required.

```bash 
docker run -d --name jaeger -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 -e COLLECTOR_OTLP_ENABLED=true -p 6831:6831/udp -p 6832:6832/udp -p 5778:5778 -p 16686:16686 -p 4317:4317 -p 4318:4318 -p 14250:14250 -p 14268:14268 -p 14269:14269 -p 9411:9411 jaegertracing/all-in-one:1.44 
``` 

The trace server runs on http://localhost:16686

# Use the system through Command Line 

An example run is 

```bash 
python run_image_classification.py --task image_classification --agent tensorflow --model_name ResNet_v1_101 --architecture gpu --num_warmup 2 --dataset_name test --dataset_path ./test_data --batch_size 2
``` 
