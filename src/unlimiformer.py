# this is adapted from the official unlimiformer repo:
# https://github.com/abertsch72/unlimiformer/blob/main/src/unlimiformer.py
# changes, comments and additions by StrangeTcy


import logging
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from transformers import (BartModel, 
                          BartForConditionalGeneration, 
                          T5Model, 
                          T5ForConditionalGeneration, 
                          LEDModel, 
                          LEDForConditionalGeneration, 
                          LlamaModel,
                          LlamaForCausalLM, 
                          LlamaForCausalLMWithFlashAttn,
                          AutoModelForSeq2SeqLM)

from typing import TypeVar, Generic

from models.index_building import Datastore, DatastoreBatch

from tqdm.auto import tqdm, trange

# for sys.exit
import sys



logger = logging.getLogger('attention_knn')
logger.setLevel(20)

ModelType = TypeVar('ModelType')





class ActivationCapturer(nn.Module):
    def __init__(self, 
                layer, 
                capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    
    def forward(self, 
                module, 
                input, 
                output):
        self.captured = input if self.capture_input else output
        if not self.layer.training:
            self.captured = self.captured.detach()




# I have no idea what Generics and TypeVars do
# TODO: figure out
class Unlimiformer(Generic[ModelType]):
    def __init__(self, 
            model: ModelType, 
            layer_begin=-1, 
            layer_end=None,
            unlimiformer_head_num=None, 
            normalize=False, 
            exclude_attention=False, 
            model_encoder_max_len=None,
            chunk_overlap=0,
            verbose=False, 
            save_heatmap=False, 
            tokenizer=None, 
            unlimiformer_training=False,
            use_datastore=False, 
            flat_index=False,
            test_datastore=False, 
            reconstruct_embeddings=False, 
            gpu_datastore=False, 
            gpu_index=False,
            knn_training=False,):
        
        print("Actual real Unlimiformer.__init__ was just called...")

        self.model = model
        self.layer_begin = layer_begin
        self.layer_end = layer_end
        self.specific_head = unlimiformer_head_num
        self.normalize = normalize
        self.exclude_attention = exclude_attention
        self.actual_model_window_size = None
        self.model_encoder_max_len = model_encoder_max_len
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self.save_heatmap = save_heatmap
        self.tokenizer = tokenizer
        self.unlimiformer_training = unlimiformer_training

        self.use_datastore = use_datastore
        self.flat_index = flat_index
        self.reconstruct_embeddings = reconstruct_embeddings
        self.gpu_datastore = gpu_datastore
        self.gpu_index = gpu_index
        self.test_datastore = test_datastore # flag for debugging

        self.knn_training = knn_training


        # this might not scale well 
        # to a multi-GPU env
        # TODO: be on the lookout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_capturer = None
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.hook_handles = []
        self.is_input_encoding_pass = False
        self.is_first_test_decoding_step = False
        self.prev_tokens = None
        self.last_beam_idx = None
        self.heatmap = None
        self.cur_decoder_layer_index = None
        self.datastore = None

        print("calling self.break_into...")
        self.break_into(model)


    # this is copied from UnlimiformerBart
    def attention_layer_to_run(self, layer_begin, layer_end): 
        print(f"our model is {self.model.base_model}")
        # input("ok?")
        return self.model.base_model.layers[layer_begin:layer_end]


    # this is copied from UnlimiformerBart
    def self_attention(self, decoder_layer):
        return decoder_layer.self_attn 


    # this is copied from UnlimiformerBart
    # and adapted for LLaMA
    def cross_attention(self, decoder_layer):
        #print(f"cross_attention got layer {decoder_layer}")
        # return decoder_layer.encoder_attn
        return decoder_layer.self_attn
    

    # this is copied from UnlimiformerBart
    # and adapted for LLaMA
    def attention_op_to_run(self, layer_begin, layer_end):
        # return [
        #         layer.encoder_attn.q_proj
        #             for layer 
        #             in self.model.base_model.decoder.layers[layer_begin:layer_end]
        #     ]
    
        return [
            layer.self_attn.q_proj
                for layer 
                in self.model.base_model.layers[layer_begin:layer_end]
        ]


    # this is copied
    # from UnlimiformerBart
    def create_key_value(self, 
                        hidden_states,
                        encoder_hidden_states, 
                        decoder_layer):
        # (batch, time, hidden_dim)
        
        # TODO: figure out the correct names:
        # this is called encoder_attn in Bart
        # and layer[1].EncDecAttention in T5
        # print(f"our decoder_layer is {decoder_layer}")
        # input("cool?")
        # attention = decoder_layer.encoder_attn
        
        # LLaMA has no encoder_attn since it has no encoders
        #  but it **does** have self_attn
        attention = decoder_layer.self_attn
        
        
        # key, value: (batch, heads, time, attn_dim)
        # TODO: k_proj vs k

        # what we have here as encoder_hidden_states
        # is None

        if self.is_encoder_decoder:
            key = attention.k_proj(encoder_hidden_states)
        else:
            print(f"Well, we're using a model that is not encoder-decoder: \nself.is_encoder_decoder is {self.is_encoder_decoder}")
            key = attention.k_proj(hidden_states)



        # TODO: num_heads vs n_heads, 
        # head_dim vs key_value_proj_dim
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        
        # TODO: v_proj vs v
        
        if self.is_encoder_decoder:
            value = attention.v_proj(encoder_hidden_states)
        else:
            print(f"Well, we're using a model that is not encoder-decoder: \nself.is_encoder_decoder is {self.is_encoder_decoder}")
            value = attention.v_proj(hidden_states)
        
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()

        # key, value: (batch, heads, time, attn_dim)
        return key, value 


    # this is copied
    # from UnlimiformerBart
    def create_decoder_layer_args(self, 
                                hidden_states, 
                                attention_mask, 
                                encoder_hidden_states,
                                encoder_attention_mask, 
                                layer_head_mask, 
                                cross_attn_layer_head_mask,
                                past_key_value, 
                                output_attentions, 
                                position_bias,
                                encoder_decoder_position_bias, 
                                use_cache, 
                                key, 
                                value):
        
        # TODO: position_bias,
        # encoder_decoder_position_bias ??
        args = {'hidden_states': hidden_states, 
                'attention_mask': attention_mask, 
                'encoder_hidden_states': encoder_hidden_states, 
                'encoder_attention_mask': encoder_attention_mask, 
                'layer_head_mask': layer_head_mask, 
                'cross_attn_layer_head_mask': cross_attn_layer_head_mask, 
                'past_key_value': (None, None, key, value), 
                'output_attentions': output_attentions, 
                'use_cache': use_cache,}
        
        if self.is_encoder_decoder:
            print("in create_decoder_layer_args, removing unnecessary items from the args dict...")
            args.pop("encoder_hidden_states", None)

        if key is None and value is None:
            args['past_key_value'] = None
        return args
    

    # this is copied
    # from UnlimiformerBart
    def process_query(self, output):
        # (batch, time, heads, attn_dim)
        # attention = self.model.base_model.decoder.layers[-1].encoder_attn
        attention = self.model.base_model.layers[-1].self_attn

        # query: (batch, heads, time, attn_dim)
        # query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).contiguous()
        return query



    # this is copied
    # from UnlimiformerBart
    def attention_layer_to_capture(self, layer_begin, layer_end): 
        if self.is_encoder_decoder:
            magic_list = [
            [layer.encoder_attn.k_proj, layer.encoder_attn.v_proj]
            for layer 
            in self.model.base_model.decoder.layers[layer_begin:layer_end]
            ]
        else:
            magic_list = [
            [layer.self_attn.k_proj, layer.self_attn.v_proj]
            for layer 
            in self.model.base_model.layers[layer_begin:layer_end]
            ]

        return magic_list 






    def break_into(self, model):
        print("Unlimiformer.break_into was just called...")
        print("setting model window size...")
        self.actual_model_window_size = self.window_size()
        if self.model_encoder_max_len is None:
            self.model_encoder_max_len = self.actual_model_window_size
        self.window_margin = int(self.model_encoder_max_len * self.chunk_overlap / 2)
        self.num_heads = model.config.num_attention_heads
        print(f"sanity check: self.num_heads is {self.num_heads}")
        if self.specific_head is None:
            # fancy
            # see https://realpython.com/python-ellipsis/
            # for an explanation
            self.head_nums = Ellipsis # torch.arange(0, self.num_heads, device=self.device)
        else:
            self.head_nums = self.specific_head
        # Save a reference to the wrapper
        model.knn_wrapper = self
        self.hooks_injected = False
        self.training_hooks_injected = False
        print("at the moment no hooks are injected...")
        self.original_forward_func = model.forward

        # Activate AttentionKNN when calling model.eval(), 
        # deactivate for model.train()
        print("setting our pre_***_hooks:")
        self.original_model_eval_func = model.eval
        print("model.eval = self.pre_eval_hook...")
        model.eval = self.pre_eval_hook
        print("explicitly calling model.eval from Unlimiformer.break_into...")
        model.eval()
        # input("cool?")

        self.original_model_train_func = model.train
        print("model.train = self.pre_train_hook...")
        model.train = self.pre_train_hook


    def pre_eval_hook(self):
        print("pre_eval_hook was called...")

        print("removing training hooks...")
        self.remove_training_hooks(self.model)

        print("and now we're injecting hooks...")
        self.inject_hooks(self.model)

        print("calling self.original_model_eval_func(). For some reason...")
        self.original_model_eval_func()


    def pre_train_hook(self, mode=True):
        # TODO: possibly switch to explicit use of
        # "train"/"eval" instead of True/False

        # mode=True means model.train() is called
        # mode=False means model.eval() is called
        print(f"Unlimiformer.pre_train_hook was just called, and mode was {mode}...")
        torch.cuda.empty_cache()
        if mode is True:
            print("calling self.break_out...")
            self.break_out(self.model)
            if self.unlimiformer_training:
                print("calling self.inject_training_hooks")
                self.inject_training_hooks(self.model)
        self.original_model_train_func(mode)
        

    def inject_hooks(self, model):
        print("inject_hooks was called, injecting these hooks!")
        print("🪝"*5)
        print("---===|||===---")


        if self.hooks_injected:
            return
        # Inject our activation_capturer 
        # to capture the activations at every forward pass
        
        attention_layers_to_capture = self.attention_layer_to_capture(self.layer_begin, self.layer_end)
        print(f"and now our attention_layers_to_capture are {attention_layers_to_capture}")
        self.activation_capturer = []
        print("injecting our activation_capturer...")
        # input("ok?")
        for layer in tqdm(attention_layers_to_capture):
            if type(layer) is list:
                layer_capturers = []
                for k_or_v in layer:
                    capturer = ActivationCapturer(k_or_v, capture_input=False)
                    layer_capturers.append(capturer)
                    self.register_hook(k_or_v, capturer)
                self.activation_capturer.append(layer_capturers)
            else:
                capturer = ActivationCapturer(layer, capture_input=False)
                self.register_hook(attention_layers_to_capture, capturer)
                self.activation_capturer.append(capturer)

        # Inject our main function 
        # after the main attention function
        attention_layers_to_run = self.attention_op_to_run(self.layer_begin, self.layer_end)
        for layer in tqdm(attention_layers_to_run):
            print("registering attention forward hook...")
            self.register_hook(layer, self.attention_forward_hook)

        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        self.original_decoder_layer_cross_attn_forward_funcs = []
        for i, decoder_layer in enumerate(decoder_layers_to_run):
            self.original_decoder_layer_cross_attn_forward_funcs.append(self.cross_attention(decoder_layer).forward)
            self.cross_attention(decoder_layer).forward = self.create_cross_attn_pre_forward_hook(self.cross_attention(decoder_layer).forward, decoder_layer, i)

        # Inject our hook function in the beginning of generation.
        # When the "model.generate()" will be called, 
        # it will first call our "reset_generation()" function, 
        # and only then call "model.generate()"
        self.original_generate_func = model.generate
        print("replacing the pre_generate hook...")
        model.generate = self.pre_generate_hook

        print("replacing the pre_forward hook...")
        model.forward = self.pre_forward_hook
        

        self.original_reorder_cache_func = model._reorder_cache
        print("replacing the reorder_cache hook...")
        model._reorder_cache = self.reorder_cache_hook
        
        print("And now self.hooks_injected = True!")
        print("---===|||===---")
        self.hooks_injected = True


    def inject_training_hooks(self, model):
        print("Unlimiformer.inject_training_hooks was just called...")
        if self.training_hooks_injected:
            return
        # self.original_forward_func = model.forward
        print("setting model.forward = self.pre_forward_hook...")
        model.forward = self.pre_forward_hook

        print("calling self.attention_layer_to_run...")
        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        
        self.original_decoder_layer_self_attn_forward_funcs = []
        for decoder_layer in tqdm(decoder_layers_to_run):
            attention = self.self_attention(decoder_layer)
            self.original_decoder_layer_self_attn_forward_funcs.append(attention.forward)
            print("calling create_self_attn_pre_forward_hook...")
            attention.forward = self.create_self_attn_pre_forward_hook(attention.forward)

        self.original_decoder_layer_cross_attn_forward_funcs = []
        for i, decoder_layer in tqdm(enumerate(decoder_layers_to_run)):
            attention = self.cross_attention(decoder_layer)
            self.original_decoder_layer_cross_attn_forward_funcs.append(attention.forward)
            print("calling create_cross_attn_pre_forward_hook...")
            attention.forward = self.create_cross_attn_pre_forward_hook(attention.forward, decoder_layer, i)

        self.original_decoder_layer_forward_funcs = []
        for decoder_layer in decoder_layers_to_run:
            self.original_decoder_layer_forward_funcs.append(decoder_layer.forward)
            print("calling create_decoder_layer_func...")
            decoder_layer.forward = self.create_decoder_layer_func(decoder_layer.forward, decoder_layer)

        print("calling self.inject_hooks_for_unaffected_layers...")
        self.inject_hooks_for_unaffected_layers(model, decoder_layers_to_run)

        print("defining our attention_layers_to_run as self.attention_op_to_run...")
        attention_layers_to_run = self.attention_op_to_run(self.layer_begin, self.layer_end)
        for layer in tqdm(attention_layers_to_run):
            print("calling self.register_hook for our train attention forward hooks...")
            self.register_hook(layer, self.train_attention_forward_hook)

        print("And now self.training_hooks_injected = True")
        self.training_hooks_injected = True


    def inject_hooks_for_unaffected_layers(self, 
                                           model, 
                                           decoder_layers_to_run):
        self.original_non_injected_decoder_layer_forward_funcs = []
        non_injected_decoder_layers = [l for l in self.attention_layer_to_run(0, None) 
            if l not in decoder_layers_to_run]
        for decoder_layer in non_injected_decoder_layers:
            self.original_non_injected_decoder_layer_forward_funcs.append(decoder_layer.forward)
            decoder_layer.forward = self.create_noninjected_decoder_layer_func(decoder_layer.forward, decoder_layer)


    def create_self_attn_pre_forward_hook(self, original_self_attn_forward_func):
        
        print("create_self_attn_pre_forward_hook was just called...")

        def self_attention_pre_forward_hook(*args, **kwargs):
            kwargs['past_key_value'] = None
            return original_self_attn_forward_func(*args, **kwargs)
        
        return self_attention_pre_forward_hook


    def create_decoder_layer_func(self, 
                                  decoder_layer_original_forward_func, 
                                  decoder_layer):
        def checkpointed_decoder_layer(
                hidden_states: torch.Tensor,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                output_attentions=False,
                position_bias=None,
                encoder_decoder_position_bias=None,
                use_cache=True):

           
            def forward_with_all_keys(hidden_states, 
                                      attention_mask, 
                                      encoder_hidden_states, 
                                      encoder_attention_mask, 
                                      layer_head_mask, 
                                      cross_attn_layer_head_mask, 
                                      past_key_value, 
                                      output_attentions, 
                                      use_cache, 
                                      long_inputs, 
                                      long_inputs_mask,
                                      position_bias, 
                                      encoder_decoder_position_bias):
                
                key, value = self.create_key_value(hidden_states=hidden_states,
                                                   encoder_hidden_states=long_inputs, 
                                                   decoder_layer=decoder_layer)
                decoder_layer_args = self.create_decoder_layer_args(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    position_bias=position_bias,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    use_cache=use_cache,
                    key=key,
                    value=value)

                # print(f"and here are our decoder_layer_args: {decoder_layer_args}")

                print("attempting another cleanup...")
                if not self.is_encoder_decoder:
                    print("removing unnecessary items from the args dict...")
                    decoder_layer_args.pop("encoder_hidden_states", None)
                    decoder_layer_args.pop("encoder_attention_mask", None)
                    decoder_layer_args.pop("layer_head_mask", None)
                    decoder_layer_args.pop("cross_attn_layer_head_mask", None)
                    #del decoder_layer_args["encoder_hidden_states"]

                    print(f"we now have these keys: {decoder_layer_args.keys()}")
                    if "encoder_hidden_states" in decoder_layer_args.keys():
                        print("What the hell?")
                        sys.exit(1)

                return decoder_layer_original_forward_func(**decoder_layer_args)

            return torch.utils.checkpoint.checkpoint(
                forward_with_all_keys, 
                hidden_states, 
                attention_mask, 
                encoder_hidden_states, 
                encoder_attention_mask, 
                layer_head_mask, 
                cross_attn_layer_head_mask, 
                None, 
                output_attentions, 
                use_cache, 
                self.long_inputs_encoded, 
                self.long_inputs_mask,
                position_bias, 
                encoder_decoder_position_bias)

        return checkpointed_decoder_layer

    def create_noninjected_decoder_layer_func(self, 
                                              decoder_layer_original_forward_func, 
                                              decoder_layer):
        def checkpointed_decoder_layer(
                hidden_states: torch.Tensor,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                output_attentions=False,
                position_bias=None,
                encoder_decoder_position_bias=None,
                use_cache=True):

           
            def forward_with_all_keys(hidden_states, 
                                      attention_mask, 
                                      encoder_hidden_states, 
                                      encoder_attention_mask, 
                                      layer_head_mask, 
                                      cross_attn_layer_head_mask, 
                                      past_key_value, 
                                      output_attentions, 
                                      use_cache, 
                                      long_inputs, 
                                      long_inputs_mask,
                                      position_bias, 
                                      encoder_decoder_position_bias):
                                
                decoder_layer_args = self.create_decoder_layer_args(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    position_bias=position_bias,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    use_cache=use_cache, key=None, value=None)
                return decoder_layer_original_forward_func(**decoder_layer_args)

            return torch.utils.checkpoint.checkpoint(
                forward_with_all_keys, 
                hidden_states, 
                attention_mask, 
                encoder_hidden_states, 
                encoder_attention_mask, 
                layer_head_mask, 
                cross_attn_layer_head_mask, 
                None, 
                output_attentions, 
                use_cache, 
                self.long_inputs_encoded, 
                self.long_inputs_mask,
                position_bias, 
                encoder_decoder_position_bias)

        return checkpointed_decoder_layer


    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)


    def break_out(self, model):

        print("break_out was just called...")

        self.prompt_keys = []
        self.prompt_values = []
        self.prompt_attention_mask = []
        self.generated_input_ids = []
        torch.cuda.empty_cache()
        if not self.hooks_injected:
            return

        for h in self.hook_handles:
            h.remove()
        model.generate = self.original_generate_func
        model.forward = self.original_forward_func
        model._reorder_cache = self.original_reorder_cache_func

        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_cross_attn_forward_funcs):
            self.cross_attention(decoder_layer).forward = original_func
        
        print("and now self.hooks_injected = False")
        
        self.hooks_injected = False


    def remove_training_hooks(self, model):

        print("removing training hooks...")

        self.long_inputs_encoded, self.long_inputs_mask = None, None
        if not self.training_hooks_injected:
            return
        for h in self.hook_handles:
            h.remove()
        model.forward = self.original_forward_func

        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_self_attn_forward_funcs):
            self.self_attention(decoder_layer).forward = original_func
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_cross_attn_forward_funcs):
            self.cross_attention(decoder_layer).forward = original_func
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_forward_funcs):
            decoder_layer.forward = original_func

        non_injected_decoder_layers = [l for l in self.attention_layer_to_run(0, None) 
            if l not in decoder_layers_to_run]
        for decoder_layer, original_func in zip(non_injected_decoder_layers, self.original_non_injected_decoder_layer_forward_funcs):
            decoder_layer.forward = original_func


        print("and now self.training_hooks_injected = False")
        self.training_hooks_injected = False


    def reset_memory(self, 
                     input_ids, 
                     attention_mask):
        
        print("reset memory was called...")

        if self.use_datastore:
            self.datastore = DatastoreBatch(dim=self.model.config.hidden_size, 
                                            batch_size=input_ids.shape[0], 
                                            flat_index=self.flat_index, 
                                            gpu_index=self.gpu_index)
            self.embeddings = []
            torch.cuda.empty_cache()
        self.prompt_input_ids = input_ids
        self.input_ids = torch.tensor([], 
                                      dtype=torch.long, 
                                      device=input_ids.device)
        self.prompt_keys, self.prompt_values = None, None
        self.prev_tokens = [None for _ in range(len(self.original_decoder_layer_cross_attn_forward_funcs))]
        self.last_beam_idx = None
        print("self.cur_layer_key_value_placeholder is set to `None`...")
        self.cur_layer_key_value_placeholder = None
        self.is_input_encoding_pass = True
        dummy_labels = torch.zeros((input_ids.shape[0], 1), 
                                   dtype=torch.long, 
                                   device=input_ids.device)
        if self.save_heatmap:
            if self.heatmap is not None:
                print(f'Generated: {self.tokenizer.decode(self.generated_input_ids[0])}')
                self.plot_heatmap(self.heatmap[0].detach().cpu().numpy())
            self.heatmap = torch.tensor([], 
                                    dtype=torch.float, 
                                    device=input_ids.device)
        self.generated_input_ids = torch.tensor([], 
                                    dtype=torch.long, 
                                    device=input_ids.device)

        self.prompt_keys = []
        self.prompt_values = []
        self.prompt_attention_mask = []
        window_indices = self.window_indices(input_ids.shape[-1])

        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in window_indices:
            chunk = input_ids[:, context_start_ind:context_end_ind]
            chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind]
            hidden_states = self.model(chunk, 
                                attention_mask=chunk_attention_mask, 
                                labels=dummy_labels, 
                                return_dict=True)
            last_hidden = hidden_states.encoder_last_hidden_state # (batch, chunked_source_len, dim)
            if self.use_datastore:
                to_add = last_hidden[:, update_start_ind:update_end_ind].detach()
                to_apply_mask = chunk_attention_mask[:, update_start_ind:update_end_ind]
                if not self.reconstruct_embeddings:
                    to_add_embeddings = to_add
                    if not self.gpu_datastore:
                        to_add_embeddings = to_add_embeddings.cpu()
                    self.embeddings.append(to_add_embeddings)
                # list of len batch, each item is (masked_time, dim)
                to_add = [key[mask.bool()] for key, mask in zip(to_add, to_apply_mask)]
                self.datastore.add_keys(to_add)
            if (not self.use_datastore) or self.test_datastore:
                layers_kv = [
                    self.process_key_value(layer_capturer) # (batch, head, time, dim)
                    for layer_capturer in self.activation_capturer
                ] # list of pairs of (batch, head, time, dim)

                # list of (batch, head, chunked_time, dim)
                key = [layer[0][:, :, update_start_ind:update_end_ind] for layer in layers_kv]
                value = [layer[1][:, :, update_start_ind:update_end_ind] for layer in layers_kv]
                chunk_attention_mask = chunk_attention_mask[:, update_start_ind:update_end_ind] # (batch, chunked_time)

                key = torch.stack(key, dim=0) # (num_layers, batch, head, time, dim)
                value = torch.stack(value, dim=0) # (num_layers, batch, head, time, dim)

                self.prompt_keys.append(key) # (num_layers, batch, head, chunked_source_len, dim)
                self.prompt_values.append(value) # (num_layers, batch, head, chunked_source_len, dim)
                self.prompt_attention_mask.append(chunk_attention_mask) # (batch, chunked_source_len)
        
        if self.use_datastore:
            # keys are all in datastore already!
            if not self.reconstruct_embeddings:
                self.embeddings = torch.cat(self.embeddings, axis=1)
            self.datastore.train_index()
        if (not self.use_datastore) or self.test_datastore:
            self.prompt_keys = torch.cat(self.prompt_keys, dim=-2) # (num_layers, batch, head, source_len, dim)
            self.prompt_values = torch.cat(self.prompt_values, dim=-2) # (num_layers, batch, head, source_len, dim)
            self.prompt_attention_mask = torch.cat(self.prompt_attention_mask, dim=-1) # (batch, source_len)

        if self.normalize:
            self.prompt_keys = torch.nn.functional.normalize(self.prompt_keys, dim=-1)

        self.is_input_encoding_pass = False
        if self.verbose:
            print(f'Input: '
                f'{self.tokenizer.decode(input_ids[0][:self.actual_model_window_size], skip_special_tokens=True)} ||| '
                f'{self.tokenizer.decode(input_ids[0][self.actual_model_window_size:], skip_special_tokens=True)}')
            print()


    def chunked_encode_input(self, input_ids, attention_mask):
        long_inputs_encoded = []
        long_inputs_mask = []
        window_indices = self.window_indices(input_ids.shape[-1])

        self.is_input_encoding_pass = True
        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in window_indices:
            chunk = input_ids[:, context_start_ind:context_end_ind]
            chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind]
            output = self.model.base_model.encoder(chunk, 
                                                attention_mask=chunk_attention_mask, 
                                                return_dict=True, 
                                                output_hidden_states=True)
            encoder_last_hidden_state = output.last_hidden_state # (batch, time, dim)
            
            # list of (batch, head, chunked_time, dim)
            encoder_last_hidden_state = encoder_last_hidden_state[:, update_start_ind:update_end_ind] # (batch, chunked_time, dim)
            chunk_attention_mask = chunk_attention_mask[:, update_start_ind:update_end_ind] # (batch, chunked_time)

            long_inputs_encoded.append(encoder_last_hidden_state) # (batch, chunked_source_len, dim)
            long_inputs_mask.append(chunk_attention_mask) # (batch, chunked_source_len)
        
        long_inputs_encoded = torch.cat(long_inputs_encoded, dim=1) # (batch, source_len, dim)
        long_inputs_mask = torch.cat(long_inputs_mask, dim=1) # (batch, source_len)

        self.is_input_encoding_pass = False
        if self.verbose:
            print(f'Input: '
                f'{self.tokenizer.decode(input_ids[0][:self.actual_model_window_size], skip_special_tokens=True)} ||| '
                f'{self.tokenizer.decode(input_ids[0][self.actual_model_window_size:], skip_special_tokens=True)}')
            print()
        return long_inputs_encoded, long_inputs_mask


    def window_indices(self, total_seq_len):
        # Copied from SLED (Ivgy et al., 2022)
        # https://github.com/Mivg/SLED/blob/main/sled/modeling_sled.py#L467
        if total_seq_len <= self.model_encoder_max_len:
            return [(0, total_seq_len, 0, total_seq_len)]
        else:
            results = []
            stride = self.model_encoder_max_len - 2 * self.window_margin
            # if self.chunk_overlap == 0:
            #     stride = self.model_encoder_max_len
            context_start = update_start_ind = 0
            context_end = self.model_encoder_max_len
            update_end_ind = context_end - self.window_margin
            # first window always should update from the beginning
            results.append((context_start, context_end, update_start_ind, update_end_ind))  

            while context_end < total_seq_len:
                context_end = min(total_seq_len, context_end + stride)
                context_start = (
                    context_start + stride if context_end < total_seq_len else total_seq_len - self.model_encoder_max_len
                )
                update_start_ind = max(update_start_ind + stride, update_end_ind)
                # last window always should update until the end
                update_end_ind = (
                    min(total_seq_len, update_end_ind + stride) if context_end < total_seq_len else total_seq_len
                )

                cs, ce, us, ue = context_start, context_end, update_start_ind - context_start, \
                                 update_end_ind - context_start

                results.append((cs, ce, us, ue))
            return results


    def pre_generate_hook(self, 
                          input_ids, 
                          **kwargs):
        print("calling self.reset_memory...")
        self.reset_memory(input_ids, kwargs['attention_mask'])
        new_kwargs = kwargs
        if 'attention_mask' in kwargs:
            new_kwargs = {k: v for k, v in kwargs.items() if k != 'attention_mask'}
            new_kwargs['attention_mask'] = kwargs['attention_mask'][:, :self.actual_model_window_size]
        new_kwargs['use_cache'] = True


        print("calling original generate...")
        return self.original_generate_func(input_ids[:, :self.actual_model_window_size], **new_kwargs)


    def pre_forward_hook(self, 
                         input_ids=None, 
                         attention_mask=None, 
                         labels=None, 
                         **kwargs):
        print("Unlimiformer.pre_forward_hook was called...")
        print(f"pre_forward_hook has received kwargs {kwargs}")
        print("setting self.model.base_model.decoder.gradient_checkpointing = False")
        if self.is_encoder_decoder:
            self.model.base_model.decoder.gradient_checkpointing = False
        else:
            self.model.base_model.gradient_checkpointing = False
        if not self.is_input_encoding_pass:
            if self.model.training:
                # self.reset_memory(input_ids, attention_mask)
                self.long_inputs_encoded, self.long_inputs_mask = self.chunked_encode_input(input_ids=input_ids, attention_mask=attention_mask)
                input_ids = input_ids[:, :self.actual_model_window_size]
                attention_mask = attention_mask[:, :self.actual_model_window_size] if attention_mask is not None else None
                # input_ids = input_ids[:, :self.model_encoder_max_len]
                # labels = labels[:, :self.model_encoder_max_len] if labels is not None else None
            else:
                if kwargs.get('past_key_values') is None:
                    self.is_first_test_decoding_step = True

                if input_ids is not None:
                    self.input_ids = torch.cat([self.input_ids, input_ids[0]])
                if kwargs.get('decoder_input_ids') is not None:
                    self.generated_input_ids = torch.cat([self.generated_input_ids, kwargs['decoder_input_ids']], axis=-1)
            
        print("calling our self.original_forward_func...")    
        input(f"our original_forward_func is {self.original_forward_func}, \nand we'll be passing {kwargs} to it...")
        print("@@@@@@@@@@@@@@@@@@@@@@\n"*4)
        input("is this right?")
        result = self.original_forward_func(input_ids=input_ids, 
                                            labels=labels, 
                                            attention_mask=attention_mask, 
                                            **kwargs)
        self.is_first_test_decoding_step = False
        print("...and returning result...")
        return result


    def create_cross_attn_pre_forward_hook(self, original_cross_attn_forward_func, decoder_layer, i):
        
        print("create_cross_attn_pre_forward_hook was just called...")
        
        def attention_pre_forward_hook(hidden_states, attention_mask=None, *args, **kwargs):
            self.cur_decoder_layer_index = i
            if kwargs.get('past_key_value') is not None:
                # it's a tuple, 
                # and we convert it to a list 
                # to be able to perform assignment 
                # and modify its items 
                # from our attention_forward_hook
                self.cur_layer_key_value_placeholder = \
                    kwargs['past_key_value'] = list(kwargs['past_key_value']) # (batch, head, time, attn_dim)

            if self.model.training:
                # from: (batch, tgt_len, dim) to: (batch * tgt_len, 1, dim)
                batch_size, tgt_len, dim = hidden_states.shape
                hidden_states = hidden_states.reshape(-1, 1, hidden_states.shape[-1])
                # from: (batch, 1, tgt_len, dim) to: (batch * tgt_len, 1, 1, dim)
                attention_mask = attention_mask.reshape(-1, 1, 1, attention_mask.shape[-1])
                
                attn_output, attn_weights_reshaped, past_key_value = original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
                attn_output = attn_output.reshape(batch_size, tgt_len, dim)
                return attn_output, attn_weights_reshaped, past_key_value
            else:
                return original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
        
        return attention_pre_forward_hook


    # this is where the magic happens
    def attention_forward_hook(self, 
                               module, 
                               input, 
                               output):
        
        print("attention_forward_hook was just called...")

        # output: (batch, time, 3 * heads * attention_dim)
        if self.is_input_encoding_pass \
            or self.is_first_test_decoding_step:
            print("in attention_forward_hook, returning...")
            return
        with torch.no_grad():
            query = self.process_query(output)[:,-1] # (batch * beam, head, dim)
            query = query[:, self.head_nums] # (batch * beam, head, dim)
            if self.normalize:
                query = torch.nn.functional.normalize(query, dim=-1)

            if self.use_datastore:
                # query: (batch, beam, head, dim)
                # need to multiply by key vector
                # query.view(query.shape[0], query.shape[1] * query.shape[2])
                # k_proj in attention? 
                attention_layer_list = self.attention_layer_to_capture(self.layer_begin, self.layer_end)
                k_proj_layer = [layers[0] for layers in attention_layer_list][self.cur_decoder_layer_index]
                v_proj_layer = [layers[1] for layers in attention_layer_list][self.cur_decoder_layer_index]
                
                # modify query by k_projs 
                k_proj = k_proj_layer.weight
                k_proj = k_proj.view(1, self.num_heads, query.shape[-1], k_proj.shape[0]) # (1, num_heads, attn_dim, embed_dim)
                datastore_query = query.unsqueeze(-2) # (batch * beam, num_heads, 1, attn_dim)
                datastore_query = torch.matmul(datastore_query, k_proj) # (batch * beam, num_heads, 1, embed_dim)
                datastore_query = datastore_query.squeeze(-2)  # (batch * beam, num_heads, embed_dim)
                datastore_query = datastore_query.view((self.datastore.batch_size, -1, datastore_query.shape[2])) # (batch, beam * num_heads, embed_dim)
                # then search
                if self.reconstruct_embeddings:
                    # embeddings: (batch, beam * head, actual_model_window_size, dim)
                    top_search_key_scores, top_search_key_indices, embeddings = self.datastore.search_and_reconstruct(datastore_query, k=self.actual_model_window_size) 
                else:
                    top_search_key_scores, top_search_key_indices = self.datastore.search(datastore_query, k=self.actual_model_window_size)
                    # self.embeddings: (batch,              src_len, dim)
                    # indices:         (batch, beam * head, actual_model_window_size)
                    # embeddings: (batch, beam * head, actual_model_window_size, dim)
                    embeddings = torch.take_along_dim(input=self.embeddings.unsqueeze(1), 
                        indices=top_search_key_indices.unsqueeze(-1).to(self.embeddings.device), dim=-2)
                    embeddings = embeddings.to(self.device)
                # (batch, beam, head, actual_model_window_size)
                top_search_key_scores = top_search_key_scores.reshape((self.datastore.batch_size, -1, self.num_heads, self.actual_model_window_size))
                top_search_key_indices = top_search_key_indices.reshape((self.datastore.batch_size, -1, self.num_heads, self.actual_model_window_size))
                # embeddings: (batch, beam, head, actual_model_window_size, dim)
                embeddings = embeddings.reshape((self.datastore.batch_size, -1, self.num_heads, self.actual_model_window_size, embeddings.shape[-1]))
                                    
            # raw_values are actually token indices; 
            # need to look them up
            if (not self.use_datastore) or self.test_datastore:
                this_layer_prompt_keys = self.prompt_keys[self.cur_decoder_layer_index]
                this_layer_prompt_values = self.prompt_values[self.cur_decoder_layer_index]
                # query: (batch * beam, head, dim)
                batch_size = self.prompt_input_ids.shape[0]
                beam_size = query.shape[0] // batch_size
                # query: (batch, beam, head, dim)
                query = query.reshape(batch_size, beam_size, *query.shape[1:])
                # this_layer_prompt_keys: (batch, head, source_len, dim)
                # this_layer_prompt_keys.unsqueeze(1):  (batch, 1, head, source_len, dim)
                # query.unsqueeze(-1):             (batch, beam, head, dim, 1)
                # attn_weights:  (batch, beam, head, source_len)
                attn_weights = torch.matmul(this_layer_prompt_keys.unsqueeze(1)[:, :, self.head_nums], query.unsqueeze(-1)).squeeze(-1) 
                # attn_weights = torch.matmul(query.unsqueeze(-2), this_layer_prompt_keys.unsqueeze(1)[:, :, self.head_nums]).squeeze(-2) 
                prompt_attention_mask_to_add = (1 - self.prompt_attention_mask) * -1e9 # (batch, source_len)
                prompt_attention_mask_to_add = prompt_attention_mask_to_add.unsqueeze(1).unsqueeze(1)
                attn_weights += prompt_attention_mask_to_add # (batch, beam, head, source_len)
                if self.exclude_attention and attn_weights.shape[-1] > self.actual_model_window_size:
                    attn_weights[..., :self.actual_model_window_size] -= 1e9

                # target_keys, target_values, topk = self.get_target_slices(output)
                topk = min(self.actual_model_window_size, attn_weights.shape[-1])
                top_key_scores, top_key_indices = torch.topk(attn_weights, k=topk, dim=-1, sorted=True) # (batch, beam, head, trunc_source)
                if self.save_heatmap:
                    # heatrow: (beam, heads, source_len)
                    heatrow = torch.zeros([top_key_indices.shape[1], top_key_indices.shape[2], this_layer_prompt_keys.shape[-2]], dtype=torch.float)
                    heatrow = heatrow.scatter(index=top_key_indices[0], src=torch.ones_like(top_key_scores[0]), dim=-1)
                    # heatrow = torch.nn.functional.softmax(heatrow, dim=-1)
                    # self.heatmap: (beam, heads, targets, source_len)
                    self.heatmap = torch.cat([self.heatmap, heatrow.unsqueeze(-2)], axis=-2)

            if self.test_datastore:
                assert top_key_indices.shape == top_search_key_indices.shape
                assert torch.mean((top_key_indices == top_search_key_indices).float()) > 0.99

            if self.verbose:
                if self.is_encoder_decoder:
                    for i, beam in tqdm(enumerate(self.generated_input_ids)):
                        print(f'({i}) Generated: {self.tokenizer.decode(beam)}')
                else:
                    print(f'Generated: {self.tokenizer.decode(self.input_ids)}')
                print()
        
        print("this is where new_keys and new_values are formed...")
        
        if self.use_datastore:
            # k_proj_layer.weight, v_proj_layer.weight: (embed_dim, embed_dim)
            # embeddings: (batch, beam, head, encoder_len, embed_dim)
            embed_dim = embeddings.shape[-1]
            k_weight = k_proj_layer.weight.view(1, 1, self.num_heads, embed_dim // self.num_heads, embed_dim).transpose(-2,-1) # (1, 1, heads, embed_dim, attn_dim)
            k_bias = k_proj_layer.bias.view(1, self.num_heads, embed_dim // self.num_heads).unsqueeze(-2)
            v_weight = v_proj_layer.weight.view(1, 1, self.num_heads, embed_dim // self.num_heads, embed_dim).transpose(-2,-1)  # (1, heads, embed_dim, attn_dim)
            v_bias = v_proj_layer.bias.view(1, self.num_heads, embed_dim // self.num_heads).unsqueeze(-2)
            # new_keys, new_values: (batch, beam, head, encoder_len, attn_dim)
            new_keys = torch.matmul(embeddings, k_weight) + k_bias.unsqueeze(0) # (beam, head, encoder_len, embed_dim)
            new_values = torch.matmul(embeddings, v_weight) + v_bias.unsqueeze(0) # (beam, head, encoder_len, embed_dim)
        else:
            # this_layer_prompt_keys:   (batch,       head, source_len, dim)
            # top_key_indices:          (batch, beam, head, trunc_source)
            new_keys = torch.take_along_dim(this_layer_prompt_keys.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)
            new_values = torch.take_along_dim(this_layer_prompt_values.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)

        if self.test_datastore:
            correct_keys = torch.take_along_dim(this_layer_prompt_keys.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)
            correct_values = torch.take_along_dim(this_layer_prompt_values.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)
            assert correct_keys.shape == new_keys.shape
            assert correct_values.shape == new_values.shape
            assert torch.mean(torch.isclose(correct_keys, new_keys, rtol=1e-3, atol=1e-3).float()) > 0.99
            assert torch.mean(torch.isclose(correct_values, new_values, rtol=1e-3, atol=1e-3).float()) > 0.99


        print("in attention_forward_hook,  setting self.cur_layer_key_value_placeholder")
        self.cur_layer_key_value_placeholder[0] = new_keys.flatten(0, 1)
        self.cur_layer_key_value_placeholder[1] = new_values.flatten(0, 1)
        return


    def train_attention_forward_hook(self, 
                                     module, 
                                     input, 
                                     output):
        
        print("train_attention_forward_hook was just called...")


        # output: (batch, time, 3 * heads * attention_dim)
        if self.is_input_encoding_pass or \
            self.is_first_test_decoding_step:
            return
        this_layer_prompt_keys = self.cur_layer_key_value_placeholder[0]
        this_layer_prompt_values = self.cur_layer_key_value_placeholder[1]
        print(f"in train_attention_forward_hook, \nthis_layer_prompt_keys are {this_layer_prompt_keys}, \nand this_layer_prompt_values are {this_layer_prompt_values}")
        with torch.no_grad():
            query = self.process_query(output) # (batch * beam, tgt_len, head, dim)
            # query = query[:, :, self.head_nums] # (batch * beam, head, dim)
            if self.normalize:
                query = torch.nn.functional.normalize(query, dim=-1)

            # query: (batch * beam, tgt_len, head, dim)
            batch_size = this_layer_prompt_keys.shape[0]
            tgt_len = query.shape[0] // batch_size
            # query: (batch, tgt, head, dim)
            query = query.reshape(batch_size, tgt_len, *query.shape[2:])
            # this_layer_prompt_keys: (batch, head, source_len, dim)
            # this_layer_prompt_keys.unsqueeze(1):  (batch, 1, head, source_len, dim)
            # attn_weights:  (batch, tgt_len, head, 1, source_len)
            # attn_weights = torch.matmul(query.unsqueeze(-2), this_layer_prompt_keys.unsqueeze(1).permute(0,1,2,4,3))
            attn_weights = torch.matmul(this_layer_prompt_keys.unsqueeze(1), query.unsqueeze(-1)) \
                .reshape(batch_size, tgt_len, query.shape[-2], 1, this_layer_prompt_keys.shape[-2])
            # attn_weights = torch.matmul(query.unsqueeze(-2), this_layer_prompt_keys.unsqueeze(1)[:, :, self.head_nums]).squeeze(-2) 
            prompt_attention_mask_to_add = (1 - self.long_inputs_mask) * -1e9 # (batch, source_len)
            prompt_attention_mask_to_add = prompt_attention_mask_to_add.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            attn_weights += prompt_attention_mask_to_add # (batch, beam, head, source_len)

            # target_keys, target_values, topk = self.get_target_slices(output)
            topk = min(self.actual_model_window_size, attn_weights.shape[-1])
            top_key_scores, top_key_indices = torch.topk(attn_weights, k=min(topk, attn_weights.shape[-1]), dim=-1, sorted=True) # (batch, beam, head, tgt, trunc_source)

                   
        # this_layer_prompt_keys:   (batch,          head,    source_len, dim)
        # top_key_indices:          (batch, tgt_len, head, 1, trunc_source)
        new_keys = torch.take_along_dim(this_layer_prompt_keys.unsqueeze(2).unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
            dim=-2) # (batch, tgt_len, head, 1, trunc_source, attn_dim)
        new_values = torch.take_along_dim(this_layer_prompt_values.unsqueeze(2).unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
            dim=-2) # (batch, tgt_len, head, 1, trunc_source, attn_dim)
        
        # (batch * beam, head, tgt_len, trunc_source, attn_dim)
        self.cur_layer_key_value_placeholder[0] = new_keys.flatten(0, 1).squeeze(2)
        self.cur_layer_key_value_placeholder[1] = new_values.flatten(0, 1).squeeze(2)
        return


    def reorder_cache_hook(self, past, beam_idx):

        print("reorder_cache_hook was just called...")


        self.last_beam_idx = beam_idx
        self.generated_input_ids = self.generated_input_ids[beam_idx]
        for i, layer_prev_tokens in enumerate(self.prev_tokens):
            if layer_prev_tokens is not None:
                self.prev_tokens[i] = layer_prev_tokens.flatten(0, 1)[beam_idx].reshape(layer_prev_tokens.shape)
        if self.save_heatmap and self.heatmap.numel() > 0:
            self.heatmap = self.heatmap[beam_idx]
        return self.original_reorder_cache_func(past, beam_idx)
    

    # that's the method we're primarily supposed to use
    @classmethod
    def convert_model(cls, model, *args, **kwargs):
        # TODO: perhaps extend this 
        # to handle our LLaMA?
        # print(f"convert_model has just received model {model}, it has methods {dir(model)}")
        # print(f"it has name {model._get_name} and another name {model.name_or_path}")
        # input("ok?")
        if "llama" in model.name_or_path:
            # TODO: is this the best way to handle this?
            model_clone = model
        else:    
            model_clone = AutoModelForSeq2SeqLM.from_config(model.config)
            model_clone.load_state_dict(model.state_dict())
        type_to_class = {
            BartModel: UnlimiformerBART,
            BartForConditionalGeneration: UnlimiformerBART,
            T5Model: UnlimiformerT5,
            T5ForConditionalGeneration: UnlimiformerT5,
            LEDModel: UnlimiformerLED,
            LEDForConditionalGeneration: UnlimiformerLED,
            LlamaModel: UnlimiformerLlama,
            LlamaForCausalLM: UnlimiformerLlama,
            LlamaForCausalLMWithFlashAttn: UnlimiformerLlamaWithFlashAttn,
        }
        
        model_clone = type_to_class[type(model_clone)](model_clone, *args, **kwargs)
        
        print(f"we'll be returning this model: {model_clone}")
        # input("are you sure that's what you want?")
        
        return model_clone
        


    def plot_heatmap(self, 
                     data, 
                     xticklabels='auto', 
                     yticklabels='auto'):
        
        print("plotting heatmap...")


        # data: (heads, targets, source_len)
        import seaborn as sb
        import matplotlib.pyplot as plt
        # print('gat = np.array([')
        # for row in data[0]:
        #     rowstr = ', '.join([f'{x:.2f}' for x in row])
        #     print(f'    [{rowstr}],')
        # print(']')

        # sb.set(font_scale=1.5, rc={'text.usetex': True})
        for i in trange(data.shape[0]):
            fig, axes = plt.subplots(1, 1, figsize=(40, 100))
            cur_ax = axes
            axes.set_title(f'Head #{i}, length: {data.shape[2]}, target length: {data.shape[1]}')
            cur_ax = axes
            # annot = [[x for x in row] for row in data]
            ax = sb.heatmap(data[i], annot=False, fmt='.2f',
                            xticklabels=512, yticklabels=yticklabels, ax=cur_ax)
            ax.xaxis.tick_top()
            plt.savefig(f'knns_head{i}.pdf')
            # plt.savefig('gat_s10_contrast.pdf')
            plt.show()




####################################
## Specific Model Definitions below
####################################



class UnlimiformerBART(Unlimiformer[BartModel]):
    def __init__(self, model: BartModel, *args, **kwargs):
        super().__init__(model, *args, **kwargs)


    def create_key_value(self, encoder_hidden_states, decoder_layer):
        # (batch, time, hidden_dim)
        attention = decoder_layer.encoder_attn
        # key, value: (batch, heads, time, attn_dim)
        key = attention.k_proj(encoder_hidden_states)
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = attention.v_proj(encoder_hidden_states)
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        # key, value: (batch, heads, time, attn_dim)
        return key, value 


    def process_key_value(self, capturers):
        key_capturer, value_capturer = capturers
        key, value = key_capturer.captured, value_capturer.captured
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.layers[-1].encoder_attn

        # query, key, value: (batch, heads, time, attn_dim)
        # query = query.view(query.shape[0], query.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        
        return key, value


    def process_query(self, output):
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.layers[-1].encoder_attn
        # query: (batch, heads, time, attn_dim)
        # query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).contiguous()
        return query


    def attention_layer_to_capture(self, layer_begin, layer_end): 
        return [
            [layer.encoder_attn.k_proj, layer.encoder_attn.v_proj]
            for layer 
            in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]


    def attention_op_to_run(self, layer_begin, layer_end):
        return [
            layer.encoder_attn.q_proj
                for layer 
                in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]


    def attention_layer_to_run(self, layer_begin, layer_end): 
        return self.model.base_model.decoder.layers[layer_begin:layer_end]


    def self_attention(self, decoder_layer):
        return decoder_layer.self_attn 


    def cross_attention(self, decoder_layer):
        return decoder_layer.encoder_attn
    

    def window_size(self):
        return self.model.config.max_position_embeddings


    def create_decoder_layer_args(self, hidden_states, attention_mask, encoder_hidden_states,
                encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask,
                past_key_value, output_attentions, position_bias,
                encoder_decoder_position_bias, use_cache, key, value):
        args = {'hidden_states': hidden_states, 
                'attention_mask': attention_mask, 
                'encoder_hidden_states': encoder_hidden_states, 
                'encoder_attention_mask': encoder_attention_mask, 
                'layer_head_mask': layer_head_mask, 
                'cross_attn_layer_head_mask': cross_attn_layer_head_mask, 
                'past_key_value': (None, None, key, value), 
                'output_attentions': output_attentions, 
                'use_cache': use_cache,}
        if key is None and value is None:
            args['past_key_value'] = None
        return args



class UnlimiformerT5(Unlimiformer[T5Model]):
    def __init__(self, model: T5Model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def create_key_value(self, encoder_hidden_states, decoder_layer):
        # (batch, time, hidden_dim)
        attention = decoder_layer.layer[1].EncDecAttention
        # key, value: (batch, heads, time, attn_dim)
        key = attention.k(encoder_hidden_states)
        key = key.view(key.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2).contiguous()
        value = attention.v(encoder_hidden_states)
        value = value.view(value.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2).contiguous()
        
        return key, value 
    
    def process_key_value(self, capturers):
        key_capturer, value_capturer = capturers
        key, value = key_capturer.captured, value_capturer.captured
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.block[-1].layer[1].EncDecAttention

        # query, key, value: (batch, heads, time, attn_dim)
        # query = query.view(query.shape[0], query.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        key = key.view(key.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2).contiguous()
        
        return key, value

    def process_query(self, output):
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.block[-1].layer[1].EncDecAttention
        # query: (batch, heads, time, attn_dim)
        query = output.view(output.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).contiguous()
        return query

    def attention_layer_to_capture(self, layer_begin, layer_end):
        return [
            [layer.layer[1].EncDecAttention.k, layer.layer[1].EncDecAttention.v]
                for layer in self.model.base_model.decoder.block[layer_begin:layer_end]
        ]
    
    def attention_op_to_run(self, layer_begin, layer_end):
        return [
            layer.layer[1].EncDecAttention.q
                for layer in self.model.base_model.decoder.block[layer_begin:layer_end]
        ]
    
    def attention_layer_to_run(self, layer_begin, layer_end): 
        return self.model.base_model.decoder.block[layer_begin:layer_end]

    def self_attention(self, decoder_layer):
        return decoder_layer.layer[0]

    def cross_attention(self, decoder_layer):
        return decoder_layer.layer[1]

    def window_size(self):
        try:
            size = self.model.config.n_positions
        except AttributeError:
            size = 1024
        return size

    def create_decoder_layer_args(self, hidden_states, attention_mask, encoder_hidden_states,
            encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask,
            past_key_value, output_attentions, position_bias,
            encoder_decoder_position_bias, use_cache, key, value):
        args = {'hidden_states': hidden_states,
            'attention_mask': attention_mask,
            'position_bias': position_bias,
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': encoder_attention_mask,
            'encoder_decoder_position_bias': encoder_decoder_position_bias,
            'layer_head_mask': layer_head_mask,
            'cross_attn_layer_head_mask': cross_attn_layer_head_mask,
            'past_key_value': (None, None, key, value),
            'use_cache': use_cache,
            'output_attentions': output_attentions}
        if key is None and value is None:
            args['past_key_value'] = None
        return args



class UnlimiformerLED(UnlimiformerBART):
    def __init__(self, model: LEDModel, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def window_size(self):
        return self.model.config.max_encoder_position_embeddings


# this is a copy of 
# UnlimiformerBART
# this might not work out of the box
# TODO: test extensively
class UnlimiformerLlama(Unlimiformer[LlamaForCausalLM]):
    def __init__(self, model: LlamaForCausalLM, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        input("We've just called super.__init__ \
              from UnlimiformerLlama.__init__")

    def create_key_value(self, 
                        encoder_hidden_states, 
                        decoder_layer):
        # (batch, time, hidden_dim)
        
        # TODO: figure out the correct names:
        # this is called encoder_attn in Bart
        # and layer[1].EncDecAttention in T5
        attention = decoder_layer.encoder_attn
        # key, value: (batch, heads, time, attn_dim)
        # TODO: k_proj vs k
        key = attention.k_proj(encoder_hidden_states)
        # TODO: num_heads vs n_heads, 
        # head_dim vs key_value_proj_dim
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        # TODO: v_proj vs v
        value = attention.v_proj(encoder_hidden_states)
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        # key, value: (batch, heads, time, attn_dim)
        return key, value 


    def process_key_value(self, capturers):
        key_capturer, value_capturer = capturers
        key, value = key_capturer.captured, value_capturer.captured
        # (batch, time, heads, attn_dim)
        # TODO: decoder.layers vs decoder.block.layer
        # ecoder_attn vs EncDecAttention
        attention = self.model.base_model.decoder.layers[-1].encoder_attn

        # query, key, value: (batch, heads, time, attn_dim)
        # query = query.view(query.shape[0], query.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        
        return key, value


    def process_query(self, output):
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.layers[-1].encoder_attn
        # query: (batch, heads, time, attn_dim)
        # query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).contiguous()
        return query


    def attention_layer_to_capture(self, layer_begin, layer_end): 
        return [
            [layer.encoder_attn.k_proj, layer.encoder_attn.v_proj]
            for layer 
            in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]


    def attention_op_to_run(self, layer_begin, layer_end):
        return [
            layer.encoder_attn.q_proj
                for layer 
                in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]


    def attention_layer_to_run(self, layer_begin, layer_end): 
        print(f"our model is {self.model.base_model}")
        return self.model.base_model.decoder.layers[layer_begin:layer_end]


    def self_attention(self, decoder_layer):
        return decoder_layer.self_attn 


    def cross_attention(self, decoder_layer):
        return decoder_layer.encoder_attn
    

    def window_size(self):
        # TODO: max_position_embeddings vs n_positions
        return self.model.config.max_position_embeddings


    def create_decoder_layer_args(self, 
                                hidden_states, 
                                attention_mask, 
                                encoder_hidden_states,
                                encoder_attention_mask, 
                                layer_head_mask, 
                                cross_attn_layer_head_mask,
                                past_key_value, 
                                output_attentions, 
                                position_bias,
                                encoder_decoder_position_bias, 
                                use_cache, 
                                key, 
                                value):
        
        # TODO: position_bias,
        # encoder_decoder_position_bias ??
        args = {'hidden_states': hidden_states, 
                'attention_mask': attention_mask, 
                'encoder_hidden_states': encoder_hidden_states, 
                'encoder_attention_mask': encoder_attention_mask, 
                'layer_head_mask': layer_head_mask, 
                'cross_attn_layer_head_mask': cross_attn_layer_head_mask, 
                'past_key_value': (None, None, key, value), 
                'output_attentions': output_attentions, 
                'use_cache': use_cache,}
        if key is None and value is None:
            args['past_key_value'] = None
        return args



# this is a copy of 
# UnlimiformerBART
# this might not work out of the box
# TODO: test extensively
class UnlimiformerLlamaWithFlashAttn(Unlimiformer[LlamaForCausalLMWithFlashAttn]):
    def __init__(self, model: LlamaForCausalLMWithFlashAttn, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        input("We've just called super.__init__ \
              from UnlimiformerLlamaWithFlashAttn.__init__")

    def create_key_value(self, 
                        encoder_hidden_states, 
                        decoder_layer):
        # (batch, time, hidden_dim)
        
        # TODO: figure out the correct names:
        # this is called encoder_attn in Bart
        # and layer[1].EncDecAttention in T5
        attention = decoder_layer.encoder_attn
        # key, value: (batch, heads, time, attn_dim)
        # TODO: k_proj vs k
        key = attention.k_proj(encoder_hidden_states)
        # TODO: num_heads vs n_heads, 
        # head_dim vs key_value_proj_dim
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        # TODO: v_proj vs v
        value = attention.v_proj(encoder_hidden_states)
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        # key, value: (batch, heads, time, attn_dim)
        return key, value 


    def process_key_value(self, capturers):
        key_capturer, value_capturer = capturers
        key, value = key_capturer.captured, value_capturer.captured
        # (batch, time, heads, attn_dim)
        # TODO: decoder.layers vs decoder.block.layer
        # ecoder_attn vs EncDecAttention
        attention = self.model.base_model.decoder.layers[-1].encoder_attn

        # query, key, value: (batch, heads, time, attn_dim)
        # query = query.view(query.shape[0], query.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        
        return key, value


    def process_query(self, output):
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.layers[-1].encoder_attn
        # query: (batch, heads, time, attn_dim)
        # query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).contiguous()
        return query


    def attention_layer_to_capture(self, layer_begin, layer_end): 
        return [
            [layer.encoder_attn.k_proj, layer.encoder_attn.v_proj]
            for layer 
            in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]


    def attention_op_to_run(self, layer_begin, layer_end):
        return [
            layer.encoder_attn.q_proj
                for layer 
                in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]


    def attention_layer_to_run(self, layer_begin, layer_end): 
        print(f"our model is {self.model.base_model}")
        return self.model.base_model.decoder.layers[layer_begin:layer_end]


    def self_attention(self, decoder_layer):
        return decoder_layer.self_attn 


    def cross_attention(self, decoder_layer):
        return decoder_layer.encoder_attn
    

    def window_size(self):
        # TODO: max_position_embeddings vs n_positions
        return self.model.config.max_position_embeddings


    def create_decoder_layer_args(self, 
                                hidden_states, 
                                attention_mask, 
                                encoder_hidden_states,
                                encoder_attention_mask, 
                                layer_head_mask, 
                                cross_attn_layer_head_mask,
                                past_key_value, 
                                output_attentions, 
                                position_bias,
                                encoder_decoder_position_bias, 
                                use_cache, 
                                key, 
                                value):
        
        # TODO: position_bias,
        # encoder_decoder_position_bias ??
        args = {'hidden_states': hidden_states, 
                'attention_mask': attention_mask, 
                'encoder_hidden_states': encoder_hidden_states, 
                'encoder_attention_mask': encoder_attention_mask, 
                'layer_head_mask': layer_head_mask, 
                'cross_attn_layer_head_mask': cross_attn_layer_head_mask, 
                'past_key_value': (None, None, key, value), 
                'output_attentions': output_attentions, 
                'use_cache': use_cache,}
        if key is None and value is None:
            args['past_key_value'] = None
        return args
