import sys 
import os 

import logging 
from contextlib import nullcontext 

from opentelemetry import trace, context  
from opentelemetry.trace import set_span_in_context 
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator 
from opentelemetry.sdk.resources import SERVICE_NAME, Resource 
from opentelemetry.sdk.trace import TracerProvider 
from opentelemetry.sdk.trace.export import BatchSpanProcessor 
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter 

from . import datasets 
from .dataloader import DataLoader 
from .load import load 

import tensorflow as tf

# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder 
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from pycupti import CUPTI 

logger = logging.getLogger(__name__) 

class MLModelScope: 
  def __init__(self, architecture): 
    resource = Resource(attributes={
        SERVICE_NAME: "mlmodelscope-service"
    }) 
    trace.set_tracer_provider(TracerProvider(resource=resource)) 
    # https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html 
    span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint='http://localhost:4317', insecure=True)) 
    trace.get_tracer_provider().add_span_processor(span_processor) 

    self.tracer = trace.get_tracer(__name__) 
    self.prop = TraceContextTextMapPropagator() 
    self.carrier = {} 
    self.spans = {} 

    self.span = self.tracer.start_span(name="mlmodelscope") 
    self.ctx = set_span_in_context(self.span) 
    self.prop.inject(carrier=self.carrier, context=self.ctx) 

    self.architecture = architecture 
    if self.architecture == "gpu": 
      self.c = CUPTI(tracer=self.tracer, prop=self.prop, carrier=self.carrier) 
      print("CUPTI version", self.c.cuptiGetVersion()) 

    self.outputs = [] 

    return 

  def load_dataset(self, dataset_name, dataset_path, batch_size): 
    if dataset_name == "datasets": 
      raise RuntimeError(f"dataset name: {dataset_name} occurs an error, please try another name") 
    dataset_list = [dataset[:-3] for dataset in os.listdir(f'./mlmodelscope/datasets/') if (dataset[0] != '_' and dataset != 'datasets.py')] 
    if dataset_name in dataset_list: 
      print(f"{dataset_name} dataset exists") 
    else: 
      raise NotImplementedError(f"{dataset_name} dataset is not supported, the available datasets are as follows:\n{', '.join(dataset_list)}") 
    
    if not os.path.exists('./' + dataset_path): 
      raise RuntimeError(f"{dataset_path}, which is the path of {dataset_name} dataset, does not exist") 
    
    with self.tracer.start_as_current_span(dataset_name + ' dataset load') as dataset_load_span: 
      self.prop.inject(carrier=self.carrier, context=set_span_in_context(dataset_load_span)) 
      self.dataset = datasets.load(dataset_name, root=dataset_path) 
      self.batch_size = batch_size 
      self.dataloader = DataLoader(self.dataset, self.batch_size) 

    return 

  def load_model(self, task, agent, model_name): 
    if task == "image_classification": 
      pass 
    else: 
      raise NotImplementedError(f"{task} task is not supported")  

    if agent == 'pytorch': 
      import torch 
      self.no_grad = torch.no_grad 
    elif agent == 'tensorflow': 
      #raise NotImplementedError(f"{agent} agent is not supported") 
      import tensorflow
      # for layer in self.model.layers:
      #   layer.trainable = False
    elif agent == 'onnxruntime': 
      raise NotImplementedError(f"{agent} agent is not supported") 
    else: 
      raise NotImplementedError(f"{agent} agent is not supported") 
    self.agent = agent 
    
    model_list = [model[:-3] for model in os.listdir(f'./mlmodelscope/{self.agent}_agent') if model[0] != '_'] 
    if model_name in model_list: 
      print(f"{model_name} model exists") 
    else: 
      raise NotImplementedError(f"{model_name} model is not supported, the available models are as follows:\n{', '.join(model_list)}") 
    self.model_name = model_name 

    with self.tracer.start_as_current_span(self.model_name + ' model load') as model_load_span: 
      self.prop.inject(carrier=self.carrier, context=set_span_in_context(model_load_span)) 
      self.model = load(model_name=self.model_name, backend=self.agent) 
      if self.agent == 'pytorch': 
        # Making the code device-agnostic
        self.device = 'cuda' if ((self.architecture == "gpu") and torch.cuda.is_available()) else 'cpu' 
        self.model.model = self.model.model.to(self.device) 
      if self.agent == 'tensorflow':
        self.device = '/GPU:O' if ((self.architecture =="gpu") and tf.test.is_gpu_available()) else '/CPU:0'


    if self.agent == 'pytorch': 
      all_spans = {} 
      def pre_hook(layer_name): 
        def pre_hook(module, input): 
          prev_ctx = self.prop.extract(carrier=self.carrier) 
          token = context.attach(prev_ctx) 
          span = self.tracer.start_span(layer_name, context=prev_ctx) 
          self.prop.inject(carrier=self.carrier, context=set_span_in_context(span)) 
          all_spans[layer_name] = (span, token, prev_ctx) 
          trace.use_span(span) 
        return pre_hook 

      def hook(layer_name): 
        def hook(module, input, output): 
          span, token, prev_ctx = all_spans[layer_name] 
          span.end() 
          context.detach(token) 
          self.prop.inject(carrier=self.carrier, context=prev_ctx) 

          del all_spans[layer_name] 
        return hook 

      for name, layer in self.model.model.named_modules(): 
        layer_name = name + '_' + type(layer).__name__ 
        layer.register_forward_pre_hook(pre_hook(layer_name)) 
        layer.register_forward_hook(hook(layer_name)) 

    return 
    
    if self.agent == 'tensorflow':
      def pre_hook(layer_name):
        # This function will be called before the forward pass of a layer
        # You can modify the inputs of the layer here, or do any other processing you need
        print(f"Forward pass starting for layer {layer_name}")
        return inputs 
    
      def hook(layer_name):
          # This function will be called after the forward pass of a layer
          # You can modify the outputs of the layer here, or do any other processing you need
          print(f"Forward pass completed for layer {layer_name}")
          return outputs
      
      completed_spans = [] 
      open_spans = {} 
      for layer in model.model.layers:
          # Generate a unique name for the layer
          layer_name = layer.name + '_' + layer.__class__.__name__
          # Register a forward hook for the layer
          layer._call_hook = functools.partial(hook, layer_name=layer_name)
          # Register a forward pre hook for the layer
          layer._call_pre_hook = functools.partial(pre_hook, layer_name=layer_name)
      
      for span in completed_spans:
        print('name: {}, timing: {}'.format(span['name'], span['end_time'] - span['start_time']))
  
  def predict(self, num_warmup): 
    tracer = self.tracer 
    prop = self.prop 
    carrier = self.carrier 

    with tracer.start_as_current_span(self.model_name + ' start') as model_start_span: 
      prop.inject(carrier=carrier, context=set_span_in_context(model_start_span)) 
      with self.no_grad() if self.agent == 'pytorch' else nullcontext(): 
        if num_warmup > 0: 
          print('Warmup') 
          num_round = len(self.dataloader)
          if num_warmup > num_round: 
            print('Warmup Size is too big, so it is reduced to the number of batches') 
            num_warmup = num_round 

          with tracer.start_as_current_span(f"Warmup") as warmup_span: 
            prop.inject(carrier=carrier, context=set_span_in_context(warmup_span)) 
            for index, data in enumerate(DataLoader(self.dataset, self.batch_size)): 
              if index >= num_warmup: 
                print('Warmup done') 
                break 
              with tracer.start_as_current_span(f"Warmup Batch {index}"):  
                with tracer.start_as_current_span("preprocess") as preprocess_span: 
                  prop.inject(carrier=carrier, context=set_span_in_context(preprocess_span)) 
                  model_input = self.model.preprocess(data) 
                  if self.agent == 'pytorch': 
                    model_input = model_input.to(self.device)
                  if self.agent == 'tensorflow':
                    with tf.device(self.device):
                      model_input = tf.identity(model_input) 
                with tracer.start_as_current_span("predict") as predict_span: 
                  prop.inject(carrier=carrier, context=set_span_in_context(predict_span)) 
                  model_output = self.model.predict(model_input) 
                with tracer.start_as_current_span("postprocess") as postprocess_span: 
                  prop.inject(carrier=carrier, context=set_span_in_context(postprocess_span)) 
                  self.model.postprocess(model_output)

        with tracer.start_as_current_span(f"Evaluate"):  
          for index, data in enumerate(self.dataloader):
            with tracer.start_as_current_span(f"Evaluate Batch {index}"):  
              with tracer.start_as_current_span("preprocess") as preprocess_span: 
                prop.inject(carrier=carrier, context=set_span_in_context(preprocess_span)) 
                model_input = self.model.preprocess(data)
                if self.agent == 'pytorch': 
                  model_input = model_input.to(self.device) 
                if self.agent == 'tensorflow':
                    with tf.device(self.device):
                      model_input = tf.identity(model_input)
                  #model_input = model_input.to(self.device)
              with tracer.start_as_current_span("predict") as predict_span:  
                prop.inject(carrier=carrier, context=set_span_in_context(predict_span)) 
                model_output = self.model.predict(model_input) 
              with tracer.start_as_current_span("postprocess") as postprocess_span: 
                prop.inject(carrier=carrier, context=set_span_in_context(postprocess_span)) 
                self.outputs.extend(self.model.postprocess(model_output)) 

    if self.architecture == "gpu": 
      self.c.Close() 

    return self.outputs 

  def Close(self): 
    self.span.end() 
    return None 

  def setSpanContextCorrelationId(self, span, name): 
      self.spans[f'{name}'] = span 
  def removeSpanByCorrelationId(self, name): 
      del self.spans[f'{name}']
  def spanFromContextCorrelationId(self, name): 
      return self.spans[f'{name}'] 

  def startSpanFromContext(self, name): 
      prev_ctx = self.prop.extract(carrier=self.carrier)
      token = context.attach(prev_ctx) 
      span = self.tracer.start_span(name=name, context=prev_ctx) 
      ctx = set_span_in_context(span) 
      self.prop.inject(carrier=self.carrier, context=ctx) 
      self.setSpanContextCorrelationId((span, token, prev_ctx), name) 
      trace.use_span(span) 

  def endSpanFromContext(self, name): 
      span, token, prev_ctx = self.spanFromContextCorrelationId(name) 
      span.end() 
      context.detach(token) 
      self.prop.inject(carrier=self.carrier, context=prev_ctx) 
      self.removeSpanByCorrelationId(name) 
