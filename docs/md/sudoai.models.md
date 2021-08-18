# Models

## Module Content

Models module
@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)> at SUDO-AI

Models of deep learning for natural language processing (NLP).

**WARNING**: When you subclass `BasicModule`, you should
overwrite __init__() and forward().

### Examples

These examples illustrate how to use sudoai Models.

Word model:

```python
>>> model_w2w = Word2Word(vocab_src=58,vocab_target=847,hidden_size=128)
>>> model_tc = Word2Label(n_class=2,hidden_size=128,vocab_size=587)
```

Sequence model:

```python
>>> model = Seq2Label(n_class=5,vocab_size=125,hidden_size=128)
```


### class sudoai.models.BasicModule(name: str = 'default', version: str = '0.1.0')
Bases: `torch.nn.modules.module.Module`

Base class for all neural network modules.

Your models should also subclass this class.
All subclasses should overwrite __ini__() and forward().


#### name()
Model identifier.


* **Type**

    str



#### version()
Model version.


* **Type**

    str



* **Raises**

    **NotImplementedError** – When forward not overwrited.



#### T_destination()
alias of TypeVar(‘T_destination’, bound=`Mapping`[`str`, `torch.Tensor`])


#### add_module(name: str, module: Optional[torch.nn.modules.module.Module])
Adds a child module to the current module.

The module can be accessed as an attribute using the given name.


* **Parameters**

    
    * **name** (*string*) – name of the child module. The child module can be
    accessed from this module using the given name


    * **module** (*Module*) – child module to be added to the module.



#### apply(fn: Callable[[torch.nn.modules.module.Module], None])
Applies `fn` recursively to every submodule (as returned by `.children()`)
as well as self. Typical use includes initializing the parameters of a model
(see also nn-init-doc).


* **Parameters**

    **fn** (`Module` -> None) – function to be applied to each submodule



* **Returns**

    self



* **Return type**

    Module


Example:

```
>>> @torch.no_grad()
>>> def init_weights(m):
>>>     print(m)
>>>     if type(m) == nn.Linear:
>>>         m.weight.fill_(1.0)
>>>         print(m.weight)
>>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
>>> net.apply(init_weights)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
```


#### bfloat16()
Casts all floating point parameters and buffers to `bfloat16` datatype.


* **Returns**

    self



* **Return type**

    Module



#### buffers(recurse: bool = True)
Returns an iterator over module buffers.


* **Parameters**

    **recurse** (*bool*) – if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



* **Yields**

    *torch.Tensor* – module buffer


Example:

```
>>> for buf in model.buffers():
>>>     print(type(buf), buf.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```


#### children()
Returns an iterator over immediate children modules.


* **Yields**

    *Module* – a child module



#### cpu()
Moves all model parameters and buffers to the CPU.


* **Returns**

    self



* **Return type**

    Module



#### cuda(device: Optional[Union[int, torch.device]] = None)
Moves all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on GPU while being optimized.


* **Parameters**

    **device** (*int**, **optional*) – if specified, all parameters will be
    copied to that device



* **Returns**

    self



* **Return type**

    Module



#### double()
Casts all floating point parameters and buffers to `double` datatype.


* **Returns**

    self



* **Return type**

    Module



#### dump_patches(: bool = False)
This allows better BC support for `load_state_dict()`. In
`state_dict()`, the version number will be saved as in the attribute
_metadata of the returned state dict, and thus pickled. _metadata is a
dictionary with keys that follow the naming convention of state dict. See
`_load_from_state_dict` on how to use this information in loading.

If new parameters/buffers are added/removed from a module, this number shall
be bumped, and the module’s _load_from_state_dict method can compare the
version number and do appropriate changes if the state dict is from before
the change.


#### eval()
Sets the module in evaluation mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. `Dropout`, `BatchNorm`,
etc.

This is equivalent with `self.train(False)`.


* **Returns**

    self



* **Return type**

    Module



#### extra_repr()
Set the extra representation of the module

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.


#### float()
Casts all floating point parameters and buffers to float datatype.


* **Returns**

    self



* **Return type**

    Module



#### forward()
Defines the computation performed at every call.

Should be overridden by all subclasses.


* **Raises**

    **NotImplementedError** – When forward() not overwrited.



#### half()
Casts all floating point parameters and buffers to `half` datatype.


* **Returns**

    self



* **Return type**

    Module



#### load()
Load saved model with id and version.


* **Raises**

    **FileExistsError** – When model fine not exist.



#### load_state_dict(state_dict: OrderedDict[str, Tensor], strict: bool = True)
Copies parameters and buffers from `state_dict` into
this module and its descendants. If `strict` is `True`, then
the keys of `state_dict` must exactly match the keys returned
by this module’s `state_dict()` function.


* **Parameters**

    
    * **state_dict** (*dict*) – a dict containing parameters and
    persistent buffers.


    * **strict** (*bool**, **optional*) – whether to strictly enforce that the keys
    in `state_dict` match the keys returned by this module’s
    `state_dict()` function. Default: `True`



* **Returns**

    
    * **missing_keys** is a list of str containing the missing keys


    * **unexpected_keys** is a list of str containing the unexpected keys




* **Return type**

    `NamedTuple` with `missing_keys` and `unexpected_keys` fields



#### modules()
Returns an iterator over all modules in the network.


* **Yields**

    *Module* – a module in the network


**NOTE**: Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.modules()):
        print(idx, '->', m)

0 -> Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)
```


#### named_buffers(prefix: str = '', recurse: bool = True)
Returns an iterator over module buffers, yielding both the
name of the buffer as well as the buffer itself.


* **Parameters**

    
    * **prefix** (*str*) – prefix to prepend to all buffer names.


    * **recurse** (*bool*) – if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



* **Yields**

    *(string, torch.Tensor)* – Tuple containing the name and buffer


Example:

```
>>> for name, buf in self.named_buffers():
>>>    if name in ['running_var']:
>>>        print(buf.size())
```


#### named_children()
Returns an iterator over immediate children modules, yielding both
the name of the module as well as the module itself.


* **Yields**

    *(string, Module)* – Tuple containing a name and child module


Example:

```
>>> for name, module in model.named_children():
>>>     if name in ['conv4', 'conv5']:
>>>         print(module)
```


#### named_modules(memo: Optional[Set[torch.nn.modules.module.Module]] = None, prefix: str = '')
Returns an iterator over all modules in the network, yielding
both the name of the module as well as the module itself.


* **Yields**

    *(string, Module)* – Tuple of name and module


**NOTE**: Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)

0 -> ('', Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
))
1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
```


#### named_parameters(prefix: str = '', recurse: bool = True)
Returns an iterator over module parameters, yielding both the
name of the parameter as well as the parameter itself.


* **Parameters**

    
    * **prefix** (*str*) – prefix to prepend to all parameter names.


    * **recurse** (*bool*) – if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



* **Yields**

    *(string, Parameter)* – Tuple containing the name and parameter


Example:

```
>>> for name, param in self.named_parameters():
>>>    if name in ['bias']:
>>>        print(param.size())
```


#### parameters(recurse: bool = True)
Returns an iterator over module parameters.

This is typically passed to an optimizer.


* **Parameters**

    **recurse** (*bool*) – if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



* **Yields**

    *Parameter* – module parameter


Example:

```
>>> for param in model.parameters():
>>>     print(type(param), param.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```


#### register_backward_hook(hook: Callable[[torch.nn.modules.module.Module, Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[None, torch.Tensor]])
Registers a backward hook on the module.

This function is deprecated in favor of `nn.Module.register_full_backward_hook()` and
the behavior of this function will change in future versions.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_buffer(name: str, tensor: Optional[torch.Tensor], persistent: bool = True)
Adds a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm’s `running_mean`
is not a parameter, but is part of the module’s state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting `persistent` to `False`. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module’s
`state_dict`.

Buffers can be accessed as attributes using given names.


* **Parameters**

    
    * **name** (*string*) – name of the buffer. The buffer can be accessed
    from this module using the given name


    * **tensor** (*Tensor*) – buffer to be registered.


    * **persistent** (*bool*) – whether the buffer is part of this module’s
    `state_dict`.


Example:

```
>>> self.register_buffer('running_mean', torch.zeros(num_features))
```


#### register_forward_hook(hook: Callable[[...], None])
Registers a forward hook on the module.

The hook will be called every time after `forward()` has computed an output.
It should have the following signature:

```
hook(module, input, output) -> None or modified output
```

The input contains only the positional arguments given to the module.
Keyword arguments won’t be passed to the hooks and only to the `forward`.
The hook can modify the output. It can modify the input inplace but
it will not have effect on forward since this is called after
`forward()` is called.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_forward_pre_hook(hook: Callable[[...], None])
Registers a forward pre-hook on the module.

The hook will be called every time before `forward()` is invoked.
It should have the following signature:

```
hook(module, input) -> None or modified input
```

The input contains only the positional arguments given to the module.
Keyword arguments won’t be passed to the hooks and only to the `forward`.
The hook can modify the input. User can either return a tuple or a
single modified value in the hook. We will wrap the value into a tuple
if a single value is returned(unless that value is already a tuple).


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_full_backward_hook(hook: Callable[[torch.nn.modules.module.Module, Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[None, torch.Tensor]])
Registers a backward hook on the module.

The hook will be called every time the gradients with respect to module
inputs are computed. The hook should have the following signature:

```
hook(module, grad_input, grad_output) -> tuple(Tensor) or None
```

The `grad_input` and `grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of `grad_input` in
subsequent computations. `grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in `grad_input` and `grad_output` will be `None` for all non-Tensor
arguments.

**WARNING**: Modifying inputs or outputs inplace is not allowed when using backward hooks and
will raise an error.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_parameter(name: str, param: Optional[torch.nn.parameter.Parameter])
Adds a parameter to the module.

The parameter can be accessed as an attribute using given name.


* **Parameters**

    
    * **name** (*string*) – name of the parameter. The parameter can be accessed
    from this module using the given name


    * **param** (*Parameter*) – parameter to be added to the module.



#### requires_grad_(requires_grad: bool = True)
Change if autograd should record operations on parameters in this
module.

This method sets the parameters’ `requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).


* **Parameters**

    **requires_grad** (*bool*) – whether autograd should record operations on
    parameters in this module. Default: `True`.



* **Returns**

    self



* **Return type**

    Module



#### save()
Save model with id and version.


* **Returns**

    Path of saved model.



* **Return type**

    str



#### share_memory()

#### state_dict(destination=None, prefix='', keep_vars=False)
Returns a dictionary containing a whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.


* **Returns**

    a dictionary containing a whole state of the module



* **Return type**

    dict


Example:

```
>>> module.state_dict().keys()
['bias', 'weight']
```


#### to(\*args, \*\*kwargs)
Moves and/or casts the parameters and buffers.

This can be called as


#### to(device=None, dtype=None, non_blocking=False)

#### to(dtype, non_blocking=False)

#### to(tensor, non_blocking=False)

#### to(memory_format=torch.channels_last)
Its signature is similar to `torch.Tensor.to()`, but only accepts
floating point or complex `dtype\`s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:\`dtype`
(if given). The integral parameters and buffers will be moved
`device`, if that is given, but with dtypes unchanged. When
`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

**NOTE**: This method modifies the module in-place.


* **Parameters**

    
    * **device** (`torch.device`) – the desired device of the parameters
    and buffers in this module


    * **dtype** (`torch.dtype`) – the desired floating point or complex dtype of
    the parameters and buffers in this module


    * **tensor** (*torch.Tensor*) – Tensor whose dtype and device are the desired
    dtype and device for all parameters and buffers in this module


    * **memory_format** (`torch.memory_format`) – the desired memory
    format for 4D parameters and buffers in this module (keyword
    only argument)



* **Returns**

    self



* **Return type**

    Module


Examples:

```
>>> linear = nn.Linear(2, 2)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]])
>>> linear.to(torch.double)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]], dtype=torch.float64)
>>> gpu1 = torch.device("cuda:1")
>>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
>>> cpu = torch.device("cpu")
>>> linear.to(cpu)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16)

>>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
>>> linear.weight
Parameter containing:
tensor([[ 0.3741+0.j,  0.2382+0.j],
        [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
>>> linear(torch.ones(3, 2, dtype=torch.cdouble))
tensor([[0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
```


#### train(mode: bool = True)
Sets the module in training mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. `Dropout`, `BatchNorm`,
etc.


* **Parameters**

    **mode** (*bool*) – whether to set training mode (`True`) or evaluation
    mode (`False`). Default: `True`.



* **Returns**

    self



* **Return type**

    Module



#### training(: bool)

#### type(dst_type: Union[torch.dtype, str])
Casts all parameters and buffers to `dst_type`.


* **Parameters**

    **dst_type** (*type** or **string*) – the desired type



* **Returns**

    self



* **Return type**

    Module



#### xpu(device: Optional[Union[int, torch.device]] = None)
Moves all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.


* **Parameters**

    **device** (*int**, **optional*) – if specified, all parameters will be
    copied to that device



* **Returns**

    self



* **Return type**

    Module



#### zero_grad(set_to_none: bool = False)
Sets gradients of all model parameters to zero. See similar function
under `torch.optim.Optimizer` for more context.


* **Parameters**

    **set_to_none** (*bool*) – instead of setting to zero, set the grads to None.
    See `torch.optim.Optimizer.zero_grad()` for details.


## SubModule Based GRU

@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)>

Copyright (c) 2021 Aymen Jemi SUDO-AI


### class sudoai.models.gru.AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1, max_length=1000)
Bases: `torch.nn.modules.module.Module`


#### T_destination()
alias of TypeVar(‘T_destination’, bound=`Mapping`[`str`, `torch.Tensor`])


#### add_module(name: str, module: Optional[torch.nn.modules.module.Module])
Adds a child module to the current module.

The module can be accessed as an attribute using the given name.


* **Parameters**

    
    * **name** (*string*) – name of the child module. The child module can be
    accessed from this module using the given name


    * **module** (*Module*) – child module to be added to the module.



#### apply(fn: Callable[[torch.nn.modules.module.Module], None])
Applies `fn` recursively to every submodule (as returned by `.children()`)
as well as self. Typical use includes initializing the parameters of a model
(see also nn-init-doc).


* **Parameters**

    **fn** (`Module` -> None) – function to be applied to each submodule



* **Returns**

    self



* **Return type**

    Module


Example:

```
>>> @torch.no_grad()
>>> def init_weights(m):
>>>     print(m)
>>>     if type(m) == nn.Linear:
>>>         m.weight.fill_(1.0)
>>>         print(m.weight)
>>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
>>> net.apply(init_weights)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
```


#### bfloat16()
Casts all floating point parameters and buffers to `bfloat16` datatype.


* **Returns**

    self



* **Return type**

    Module



#### buffers(recurse: bool = True)
Returns an iterator over module buffers.


* **Parameters**

    **recurse** (*bool*) – if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



* **Yields**

    *torch.Tensor* – module buffer


Example:

```
>>> for buf in model.buffers():
>>>     print(type(buf), buf.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```


#### children()
Returns an iterator over immediate children modules.


* **Yields**

    *Module* – a child module



#### cpu()
Moves all model parameters and buffers to the CPU.


* **Returns**

    self



* **Return type**

    Module



#### cuda(device: Optional[Union[int, torch.device]] = None)
Moves all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on GPU while being optimized.


* **Parameters**

    **device** (*int**, **optional*) – if specified, all parameters will be
    copied to that device



* **Returns**

    self



* **Return type**

    Module



#### double()
Casts all floating point parameters and buffers to `double` datatype.


* **Returns**

    self



* **Return type**

    Module



#### dump_patches(: bool = False)
This allows better BC support for `load_state_dict()`. In
`state_dict()`, the version number will be saved as in the attribute
_metadata of the returned state dict, and thus pickled. _metadata is a
dictionary with keys that follow the naming convention of state dict. See
`_load_from_state_dict` on how to use this information in loading.

If new parameters/buffers are added/removed from a module, this number shall
be bumped, and the module’s _load_from_state_dict method can compare the
version number and do appropriate changes if the state dict is from before
the change.


#### eval()
Sets the module in evaluation mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. `Dropout`, `BatchNorm`,
etc.

This is equivalent with `self.train(False)`.


* **Returns**

    self



* **Return type**

    Module



#### extra_repr()
Set the extra representation of the module

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.


#### float()
Casts all floating point parameters and buffers to float datatype.


* **Returns**

    self



* **Return type**

    Module



#### forward(input, hidden, encoder_outputs)
Defines the computation performed at every call.

Should be overridden by all subclasses.

**NOTE**: Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.


#### half()
Casts all floating point parameters and buffers to `half` datatype.


* **Returns**

    self



* **Return type**

    Module



#### initHidden()

#### load_state_dict(state_dict: OrderedDict[str, Tensor], strict: bool = True)
Copies parameters and buffers from `state_dict` into
this module and its descendants. If `strict` is `True`, then
the keys of `state_dict` must exactly match the keys returned
by this module’s `state_dict()` function.


* **Parameters**

    
    * **state_dict** (*dict*) – a dict containing parameters and
    persistent buffers.


    * **strict** (*bool**, **optional*) – whether to strictly enforce that the keys
    in `state_dict` match the keys returned by this module’s
    `state_dict()` function. Default: `True`



* **Returns**

    
    * **missing_keys** is a list of str containing the missing keys


    * **unexpected_keys** is a list of str containing the unexpected keys




* **Return type**

    `NamedTuple` with `missing_keys` and `unexpected_keys` fields



#### modules()
Returns an iterator over all modules in the network.


* **Yields**

    *Module* – a module in the network


**NOTE**: Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.modules()):
        print(idx, '->', m)

0 -> Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)
```


#### named_buffers(prefix: str = '', recurse: bool = True)
Returns an iterator over module buffers, yielding both the
name of the buffer as well as the buffer itself.


* **Parameters**

    
    * **prefix** (*str*) – prefix to prepend to all buffer names.


    * **recurse** (*bool*) – if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



* **Yields**

    *(string, torch.Tensor)* – Tuple containing the name and buffer


Example:

```
>>> for name, buf in self.named_buffers():
>>>    if name in ['running_var']:
>>>        print(buf.size())
```


#### named_children()
Returns an iterator over immediate children modules, yielding both
the name of the module as well as the module itself.


* **Yields**

    *(string, Module)* – Tuple containing a name and child module


Example:

```
>>> for name, module in model.named_children():
>>>     if name in ['conv4', 'conv5']:
>>>         print(module)
```


#### named_modules(memo: Optional[Set[torch.nn.modules.module.Module]] = None, prefix: str = '')
Returns an iterator over all modules in the network, yielding
both the name of the module as well as the module itself.


* **Yields**

    *(string, Module)* – Tuple of name and module


**NOTE**: Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)

0 -> ('', Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
))
1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
```


#### named_parameters(prefix: str = '', recurse: bool = True)
Returns an iterator over module parameters, yielding both the
name of the parameter as well as the parameter itself.


* **Parameters**

    
    * **prefix** (*str*) – prefix to prepend to all parameter names.


    * **recurse** (*bool*) – if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



* **Yields**

    *(string, Parameter)* – Tuple containing the name and parameter


Example:

```
>>> for name, param in self.named_parameters():
>>>    if name in ['bias']:
>>>        print(param.size())
```


#### parameters(recurse: bool = True)
Returns an iterator over module parameters.

This is typically passed to an optimizer.


* **Parameters**

    **recurse** (*bool*) – if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



* **Yields**

    *Parameter* – module parameter


Example:

```
>>> for param in model.parameters():
>>>     print(type(param), param.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```


#### register_backward_hook(hook: Callable[[torch.nn.modules.module.Module, Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[None, torch.Tensor]])
Registers a backward hook on the module.

This function is deprecated in favor of `nn.Module.register_full_backward_hook()` and
the behavior of this function will change in future versions.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_buffer(name: str, tensor: Optional[torch.Tensor], persistent: bool = True)
Adds a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm’s `running_mean`
is not a parameter, but is part of the module’s state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting `persistent` to `False`. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module’s
`state_dict`.

Buffers can be accessed as attributes using given names.


* **Parameters**

    
    * **name** (*string*) – name of the buffer. The buffer can be accessed
    from this module using the given name


    * **tensor** (*Tensor*) – buffer to be registered.


    * **persistent** (*bool*) – whether the buffer is part of this module’s
    `state_dict`.


Example:

```
>>> self.register_buffer('running_mean', torch.zeros(num_features))
```


#### register_forward_hook(hook: Callable[[...], None])
Registers a forward hook on the module.

The hook will be called every time after `forward()` has computed an output.
It should have the following signature:

```
hook(module, input, output) -> None or modified output
```

The input contains only the positional arguments given to the module.
Keyword arguments won’t be passed to the hooks and only to the `forward`.
The hook can modify the output. It can modify the input inplace but
it will not have effect on forward since this is called after
`forward()` is called.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_forward_pre_hook(hook: Callable[[...], None])
Registers a forward pre-hook on the module.

The hook will be called every time before `forward()` is invoked.
It should have the following signature:

```
hook(module, input) -> None or modified input
```

The input contains only the positional arguments given to the module.
Keyword arguments won’t be passed to the hooks and only to the `forward`.
The hook can modify the input. User can either return a tuple or a
single modified value in the hook. We will wrap the value into a tuple
if a single value is returned(unless that value is already a tuple).


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_full_backward_hook(hook: Callable[[torch.nn.modules.module.Module, Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[None, torch.Tensor]])
Registers a backward hook on the module.

The hook will be called every time the gradients with respect to module
inputs are computed. The hook should have the following signature:

```
hook(module, grad_input, grad_output) -> tuple(Tensor) or None
```

The `grad_input` and `grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of `grad_input` in
subsequent computations. `grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in `grad_input` and `grad_output` will be `None` for all non-Tensor
arguments.

**WARNING**: Modifying inputs or outputs inplace is not allowed when using backward hooks and
will raise an error.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_parameter(name: str, param: Optional[torch.nn.parameter.Parameter])
Adds a parameter to the module.

The parameter can be accessed as an attribute using given name.


* **Parameters**

    
    * **name** (*string*) – name of the parameter. The parameter can be accessed
    from this module using the given name


    * **param** (*Parameter*) – parameter to be added to the module.



#### requires_grad_(requires_grad: bool = True)
Change if autograd should record operations on parameters in this
module.

This method sets the parameters’ `requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).


* **Parameters**

    **requires_grad** (*bool*) – whether autograd should record operations on
    parameters in this module. Default: `True`.



* **Returns**

    self



* **Return type**

    Module



#### share_memory()

#### state_dict(destination=None, prefix='', keep_vars=False)
Returns a dictionary containing a whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.


* **Returns**

    a dictionary containing a whole state of the module



* **Return type**

    dict


Example:

```
>>> module.state_dict().keys()
['bias', 'weight']
```


#### to(\*args, \*\*kwargs)
Moves and/or casts the parameters and buffers.

This can be called as


#### to(device=None, dtype=None, non_blocking=False)

#### to(dtype, non_blocking=False)

#### to(tensor, non_blocking=False)

#### to(memory_format=torch.channels_last)
Its signature is similar to `torch.Tensor.to()`, but only accepts
floating point or complex `dtype\`s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:\`dtype`
(if given). The integral parameters and buffers will be moved
`device`, if that is given, but with dtypes unchanged. When
`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

**NOTE**: This method modifies the module in-place.


* **Parameters**

    
    * **device** (`torch.device`) – the desired device of the parameters
    and buffers in this module


    * **dtype** (`torch.dtype`) – the desired floating point or complex dtype of
    the parameters and buffers in this module


    * **tensor** (*torch.Tensor*) – Tensor whose dtype and device are the desired
    dtype and device for all parameters and buffers in this module


    * **memory_format** (`torch.memory_format`) – the desired memory
    format for 4D parameters and buffers in this module (keyword
    only argument)



* **Returns**

    self



* **Return type**

    Module


Examples:

```
>>> linear = nn.Linear(2, 2)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]])
>>> linear.to(torch.double)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]], dtype=torch.float64)
>>> gpu1 = torch.device("cuda:1")
>>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
>>> cpu = torch.device("cpu")
>>> linear.to(cpu)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16)

>>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
>>> linear.weight
Parameter containing:
tensor([[ 0.3741+0.j,  0.2382+0.j],
        [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
>>> linear(torch.ones(3, 2, dtype=torch.cdouble))
tensor([[0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
```


#### train(mode: bool = True)
Sets the module in training mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. `Dropout`, `BatchNorm`,
etc.


* **Parameters**

    **mode** (*bool*) – whether to set training mode (`True`) or evaluation
    mode (`False`). Default: `True`.



* **Returns**

    self



* **Return type**

    Module



#### training(: bool)

#### type(dst_type: Union[torch.dtype, str])
Casts all parameters and buffers to `dst_type`.


* **Parameters**

    **dst_type** (*type** or **string*) – the desired type



* **Returns**

    self



* **Return type**

    Module



#### xpu(device: Optional[Union[int, torch.device]] = None)
Moves all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.


* **Parameters**

    **device** (*int**, **optional*) – if specified, all parameters will be
    copied to that device



* **Returns**

    self



* **Return type**

    Module



#### zero_grad(set_to_none: bool = False)
Sets gradients of all model parameters to zero. See similar function
under `torch.optim.Optimizer` for more context.


* **Parameters**

    **set_to_none** (*bool*) – instead of setting to zero, set the grads to None.
    See `torch.optim.Optimizer.zero_grad()` for details.



### class sudoai.models.gru.EncoderRNN(input_size, hidden_size)
Bases: `torch.nn.modules.module.Module`


#### T_destination()
alias of TypeVar(‘T_destination’, bound=`Mapping`[`str`, `torch.Tensor`])


#### add_module(name: str, module: Optional[torch.nn.modules.module.Module])
Adds a child module to the current module.

The module can be accessed as an attribute using the given name.


* **Parameters**

    
    * **name** (*string*) – name of the child module. The child module can be
    accessed from this module using the given name


    * **module** (*Module*) – child module to be added to the module.



#### apply(fn: Callable[[torch.nn.modules.module.Module], None])
Applies `fn` recursively to every submodule (as returned by `.children()`)
as well as self. Typical use includes initializing the parameters of a model
(see also nn-init-doc).


* **Parameters**

    **fn** (`Module` -> None) – function to be applied to each submodule



* **Returns**

    self



* **Return type**

    Module


Example:

```
>>> @torch.no_grad()
>>> def init_weights(m):
>>>     print(m)
>>>     if type(m) == nn.Linear:
>>>         m.weight.fill_(1.0)
>>>         print(m.weight)
>>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
>>> net.apply(init_weights)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
```


#### bfloat16()
Casts all floating point parameters and buffers to `bfloat16` datatype.


* **Returns**

    self



* **Return type**

    Module



#### buffers(recurse: bool = True)
Returns an iterator over module buffers.


* **Parameters**

    **recurse** (*bool*) – if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



* **Yields**

    *torch.Tensor* – module buffer


Example:

```
>>> for buf in model.buffers():
>>>     print(type(buf), buf.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```


#### children()
Returns an iterator over immediate children modules.


* **Yields**

    *Module* – a child module



#### cpu()
Moves all model parameters and buffers to the CPU.


* **Returns**

    self



* **Return type**

    Module



#### cuda(device: Optional[Union[int, torch.device]] = None)
Moves all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on GPU while being optimized.


* **Parameters**

    **device** (*int**, **optional*) – if specified, all parameters will be
    copied to that device



* **Returns**

    self



* **Return type**

    Module



#### double()
Casts all floating point parameters and buffers to `double` datatype.


* **Returns**

    self



* **Return type**

    Module



#### dump_patches(: bool = False)
This allows better BC support for `load_state_dict()`. In
`state_dict()`, the version number will be saved as in the attribute
_metadata of the returned state dict, and thus pickled. _metadata is a
dictionary with keys that follow the naming convention of state dict. See
`_load_from_state_dict` on how to use this information in loading.

If new parameters/buffers are added/removed from a module, this number shall
be bumped, and the module’s _load_from_state_dict method can compare the
version number and do appropriate changes if the state dict is from before
the change.


#### eval()
Sets the module in evaluation mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. `Dropout`, `BatchNorm`,
etc.

This is equivalent with `self.train(False)`.


* **Returns**

    self



* **Return type**

    Module



#### extra_repr()
Set the extra representation of the module

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.


#### float()
Casts all floating point parameters and buffers to float datatype.


* **Returns**

    self



* **Return type**

    Module



#### forward(input, hidden)
Defines the computation performed at every call.

Should be overridden by all subclasses.

**NOTE**: Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.


#### half()
Casts all floating point parameters and buffers to `half` datatype.


* **Returns**

    self



* **Return type**

    Module



#### initHidden()

#### load_state_dict(state_dict: OrderedDict[str, Tensor], strict: bool = True)
Copies parameters and buffers from `state_dict` into
this module and its descendants. If `strict` is `True`, then
the keys of `state_dict` must exactly match the keys returned
by this module’s `state_dict()` function.


* **Parameters**

    
    * **state_dict** (*dict*) – a dict containing parameters and
    persistent buffers.


    * **strict** (*bool**, **optional*) – whether to strictly enforce that the keys
    in `state_dict` match the keys returned by this module’s
    `state_dict()` function. Default: `True`



* **Returns**

    
    * **missing_keys** is a list of str containing the missing keys


    * **unexpected_keys** is a list of str containing the unexpected keys




* **Return type**

    `NamedTuple` with `missing_keys` and `unexpected_keys` fields



#### modules()
Returns an iterator over all modules in the network.


* **Yields**

    *Module* – a module in the network


**NOTE**: Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.modules()):
        print(idx, '->', m)

0 -> Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)
```


#### named_buffers(prefix: str = '', recurse: bool = True)
Returns an iterator over module buffers, yielding both the
name of the buffer as well as the buffer itself.


* **Parameters**

    
    * **prefix** (*str*) – prefix to prepend to all buffer names.


    * **recurse** (*bool*) – if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



* **Yields**

    *(string, torch.Tensor)* – Tuple containing the name and buffer


Example:

```
>>> for name, buf in self.named_buffers():
>>>    if name in ['running_var']:
>>>        print(buf.size())
```


#### named_children()
Returns an iterator over immediate children modules, yielding both
the name of the module as well as the module itself.


* **Yields**

    *(string, Module)* – Tuple containing a name and child module


Example:

```
>>> for name, module in model.named_children():
>>>     if name in ['conv4', 'conv5']:
>>>         print(module)
```


#### named_modules(memo: Optional[Set[torch.nn.modules.module.Module]] = None, prefix: str = '')
Returns an iterator over all modules in the network, yielding
both the name of the module as well as the module itself.


* **Yields**

    *(string, Module)* – Tuple of name and module


**NOTE**: Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)

0 -> ('', Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
))
1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
```


#### named_parameters(prefix: str = '', recurse: bool = True)
Returns an iterator over module parameters, yielding both the
name of the parameter as well as the parameter itself.


* **Parameters**

    
    * **prefix** (*str*) – prefix to prepend to all parameter names.


    * **recurse** (*bool*) – if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



* **Yields**

    *(string, Parameter)* – Tuple containing the name and parameter


Example:

```
>>> for name, param in self.named_parameters():
>>>    if name in ['bias']:
>>>        print(param.size())
```


#### parameters(recurse: bool = True)
Returns an iterator over module parameters.

This is typically passed to an optimizer.


* **Parameters**

    **recurse** (*bool*) – if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



* **Yields**

    *Parameter* – module parameter


Example:

```
>>> for param in model.parameters():
>>>     print(type(param), param.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```


#### register_backward_hook(hook: Callable[[torch.nn.modules.module.Module, Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[None, torch.Tensor]])
Registers a backward hook on the module.

This function is deprecated in favor of `nn.Module.register_full_backward_hook()` and
the behavior of this function will change in future versions.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_buffer(name: str, tensor: Optional[torch.Tensor], persistent: bool = True)
Adds a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm’s `running_mean`
is not a parameter, but is part of the module’s state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting `persistent` to `False`. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module’s
`state_dict`.

Buffers can be accessed as attributes using given names.


* **Parameters**

    
    * **name** (*string*) – name of the buffer. The buffer can be accessed
    from this module using the given name


    * **tensor** (*Tensor*) – buffer to be registered.


    * **persistent** (*bool*) – whether the buffer is part of this module’s
    `state_dict`.


Example:

```
>>> self.register_buffer('running_mean', torch.zeros(num_features))
```


#### register_forward_hook(hook: Callable[[...], None])
Registers a forward hook on the module.

The hook will be called every time after `forward()` has computed an output.
It should have the following signature:

```
hook(module, input, output) -> None or modified output
```

The input contains only the positional arguments given to the module.
Keyword arguments won’t be passed to the hooks and only to the `forward`.
The hook can modify the output. It can modify the input inplace but
it will not have effect on forward since this is called after
`forward()` is called.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_forward_pre_hook(hook: Callable[[...], None])
Registers a forward pre-hook on the module.

The hook will be called every time before `forward()` is invoked.
It should have the following signature:

```
hook(module, input) -> None or modified input
```

The input contains only the positional arguments given to the module.
Keyword arguments won’t be passed to the hooks and only to the `forward`.
The hook can modify the input. User can either return a tuple or a
single modified value in the hook. We will wrap the value into a tuple
if a single value is returned(unless that value is already a tuple).


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_full_backward_hook(hook: Callable[[torch.nn.modules.module.Module, Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[None, torch.Tensor]])
Registers a backward hook on the module.

The hook will be called every time the gradients with respect to module
inputs are computed. The hook should have the following signature:

```
hook(module, grad_input, grad_output) -> tuple(Tensor) or None
```

The `grad_input` and `grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of `grad_input` in
subsequent computations. `grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in `grad_input` and `grad_output` will be `None` for all non-Tensor
arguments.

**WARNING**: Modifying inputs or outputs inplace is not allowed when using backward hooks and
will raise an error.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_parameter(name: str, param: Optional[torch.nn.parameter.Parameter])
Adds a parameter to the module.

The parameter can be accessed as an attribute using given name.


* **Parameters**

    
    * **name** (*string*) – name of the parameter. The parameter can be accessed
    from this module using the given name


    * **param** (*Parameter*) – parameter to be added to the module.



#### requires_grad_(requires_grad: bool = True)
Change if autograd should record operations on parameters in this
module.

This method sets the parameters’ `requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).


* **Parameters**

    **requires_grad** (*bool*) – whether autograd should record operations on
    parameters in this module. Default: `True`.



* **Returns**

    self



* **Return type**

    Module



#### share_memory()

#### state_dict(destination=None, prefix='', keep_vars=False)
Returns a dictionary containing a whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.


* **Returns**

    a dictionary containing a whole state of the module



* **Return type**

    dict


Example:

```
>>> module.state_dict().keys()
['bias', 'weight']
```


#### to(\*args, \*\*kwargs)
Moves and/or casts the parameters and buffers.

This can be called as


#### to(device=None, dtype=None, non_blocking=False)

#### to(dtype, non_blocking=False)

#### to(tensor, non_blocking=False)

#### to(memory_format=torch.channels_last)
Its signature is similar to `torch.Tensor.to()`, but only accepts
floating point or complex `dtype\`s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:\`dtype`
(if given). The integral parameters and buffers will be moved
`device`, if that is given, but with dtypes unchanged. When
`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

**NOTE**: This method modifies the module in-place.


* **Parameters**

    
    * **device** (`torch.device`) – the desired device of the parameters
    and buffers in this module


    * **dtype** (`torch.dtype`) – the desired floating point or complex dtype of
    the parameters and buffers in this module


    * **tensor** (*torch.Tensor*) – Tensor whose dtype and device are the desired
    dtype and device for all parameters and buffers in this module


    * **memory_format** (`torch.memory_format`) – the desired memory
    format for 4D parameters and buffers in this module (keyword
    only argument)



* **Returns**

    self



* **Return type**

    Module


Examples:

```
>>> linear = nn.Linear(2, 2)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]])
>>> linear.to(torch.double)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]], dtype=torch.float64)
>>> gpu1 = torch.device("cuda:1")
>>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
>>> cpu = torch.device("cpu")
>>> linear.to(cpu)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16)

>>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
>>> linear.weight
Parameter containing:
tensor([[ 0.3741+0.j,  0.2382+0.j],
        [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
>>> linear(torch.ones(3, 2, dtype=torch.cdouble))
tensor([[0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
```


#### train(mode: bool = True)
Sets the module in training mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. `Dropout`, `BatchNorm`,
etc.


* **Parameters**

    **mode** (*bool*) – whether to set training mode (`True`) or evaluation
    mode (`False`). Default: `True`.



* **Returns**

    self



* **Return type**

    Module



#### training(: bool)

#### type(dst_type: Union[torch.dtype, str])
Casts all parameters and buffers to `dst_type`.


* **Parameters**

    **dst_type** (*type** or **string*) – the desired type



* **Returns**

    self



* **Return type**

    Module



#### xpu(device: Optional[Union[int, torch.device]] = None)
Moves all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.


* **Parameters**

    **device** (*int**, **optional*) – if specified, all parameters will be
    copied to that device



* **Returns**

    self



* **Return type**

    Module



#### zero_grad(set_to_none: bool = False)
Sets gradients of all model parameters to zero. See similar function
under `torch.optim.Optimizer` for more context.


* **Parameters**

    **set_to_none** (*bool*) – instead of setting to zero, set the grads to None.
    See `torch.optim.Optimizer.zero_grad()` for details.


## SubModule Based LSTM

@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)>

Copyright (c) 2021 Aymen Jemi SUDO-AI


### class sudoai.models.lstm.ExtremMutliLabelTextClassification(n_class=3714, vocab_size=30001, embedding_size=300, hidden_size=256, d_a=256, multiclass=False)
Bases: `torch.nn.modules.module.Module`


#### T_destination()
alias of TypeVar(‘T_destination’, bound=`Mapping`[`str`, `torch.Tensor`])


#### add_module(name: str, module: Optional[torch.nn.modules.module.Module])
Adds a child module to the current module.

The module can be accessed as an attribute using the given name.


* **Parameters**

    
    * **name** (*string*) – name of the child module. The child module can be
    accessed from this module using the given name


    * **module** (*Module*) – child module to be added to the module.



#### apply(fn: Callable[[torch.nn.modules.module.Module], None])
Applies `fn` recursively to every submodule (as returned by `.children()`)
as well as self. Typical use includes initializing the parameters of a model
(see also nn-init-doc).


* **Parameters**

    **fn** (`Module` -> None) – function to be applied to each submodule



* **Returns**

    self



* **Return type**

    Module


Example:

```
>>> @torch.no_grad()
>>> def init_weights(m):
>>>     print(m)
>>>     if type(m) == nn.Linear:
>>>         m.weight.fill_(1.0)
>>>         print(m.weight)
>>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
>>> net.apply(init_weights)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
```


#### bfloat16()
Casts all floating point parameters and buffers to `bfloat16` datatype.


* **Returns**

    self



* **Return type**

    Module



#### buffers(recurse: bool = True)
Returns an iterator over module buffers.


* **Parameters**

    **recurse** (*bool*) – if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



* **Yields**

    *torch.Tensor* – module buffer


Example:

```
>>> for buf in model.buffers():
>>>     print(type(buf), buf.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```


#### children()
Returns an iterator over immediate children modules.


* **Yields**

    *Module* – a child module



#### cpu()
Moves all model parameters and buffers to the CPU.


* **Returns**

    self



* **Return type**

    Module



#### cuda(device: Optional[Union[int, torch.device]] = None)
Moves all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on GPU while being optimized.


* **Parameters**

    **device** (*int**, **optional*) – if specified, all parameters will be
    copied to that device



* **Returns**

    self



* **Return type**

    Module



#### double()
Casts all floating point parameters and buffers to `double` datatype.


* **Returns**

    self



* **Return type**

    Module



#### dump_patches(: bool = False)
This allows better BC support for `load_state_dict()`. In
`state_dict()`, the version number will be saved as in the attribute
_metadata of the returned state dict, and thus pickled. _metadata is a
dictionary with keys that follow the naming convention of state dict. See
`_load_from_state_dict` on how to use this information in loading.

If new parameters/buffers are added/removed from a module, this number shall
be bumped, and the module’s _load_from_state_dict method can compare the
version number and do appropriate changes if the state dict is from before
the change.


#### eval()
Sets the module in evaluation mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. `Dropout`, `BatchNorm`,
etc.

This is equivalent with `self.train(False)`.


* **Returns**

    self



* **Return type**

    Module



#### extra_repr()
Set the extra representation of the module

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.


#### float()
Casts all floating point parameters and buffers to float datatype.


* **Returns**

    self



* **Return type**

    Module



#### forward(input)
Defines the computation performed at every call.

Should be overridden by all subclasses.

**NOTE**: Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.


#### half()
Casts all floating point parameters and buffers to `half` datatype.


* **Returns**

    self



* **Return type**

    Module



#### init_hidden(batch_size)

#### load_state_dict(state_dict: OrderedDict[str, Tensor], strict: bool = True)
Copies parameters and buffers from `state_dict` into
this module and its descendants. If `strict` is `True`, then
the keys of `state_dict` must exactly match the keys returned
by this module’s `state_dict()` function.


* **Parameters**

    
    * **state_dict** (*dict*) – a dict containing parameters and
    persistent buffers.


    * **strict** (*bool**, **optional*) – whether to strictly enforce that the keys
    in `state_dict` match the keys returned by this module’s
    `state_dict()` function. Default: `True`



* **Returns**

    
    * **missing_keys** is a list of str containing the missing keys


    * **unexpected_keys** is a list of str containing the unexpected keys




* **Return type**

    `NamedTuple` with `missing_keys` and `unexpected_keys` fields



#### modules()
Returns an iterator over all modules in the network.


* **Yields**

    *Module* – a module in the network


**NOTE**: Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.modules()):
        print(idx, '->', m)

0 -> Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)
```


#### named_buffers(prefix: str = '', recurse: bool = True)
Returns an iterator over module buffers, yielding both the
name of the buffer as well as the buffer itself.


* **Parameters**

    
    * **prefix** (*str*) – prefix to prepend to all buffer names.


    * **recurse** (*bool*) – if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



* **Yields**

    *(string, torch.Tensor)* – Tuple containing the name and buffer


Example:

```
>>> for name, buf in self.named_buffers():
>>>    if name in ['running_var']:
>>>        print(buf.size())
```


#### named_children()
Returns an iterator over immediate children modules, yielding both
the name of the module as well as the module itself.


* **Yields**

    *(string, Module)* – Tuple containing a name and child module


Example:

```
>>> for name, module in model.named_children():
>>>     if name in ['conv4', 'conv5']:
>>>         print(module)
```


#### named_modules(memo: Optional[Set[torch.nn.modules.module.Module]] = None, prefix: str = '')
Returns an iterator over all modules in the network, yielding
both the name of the module as well as the module itself.


* **Yields**

    *(string, Module)* – Tuple of name and module


**NOTE**: Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)

0 -> ('', Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
))
1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
```


#### named_parameters(prefix: str = '', recurse: bool = True)
Returns an iterator over module parameters, yielding both the
name of the parameter as well as the parameter itself.


* **Parameters**

    
    * **prefix** (*str*) – prefix to prepend to all parameter names.


    * **recurse** (*bool*) – if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



* **Yields**

    *(string, Parameter)* – Tuple containing the name and parameter


Example:

```
>>> for name, param in self.named_parameters():
>>>    if name in ['bias']:
>>>        print(param.size())
```


#### parameters(recurse: bool = True)
Returns an iterator over module parameters.

This is typically passed to an optimizer.


* **Parameters**

    **recurse** (*bool*) – if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



* **Yields**

    *Parameter* – module parameter


Example:

```
>>> for param in model.parameters():
>>>     print(type(param), param.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```


#### register_backward_hook(hook: Callable[[torch.nn.modules.module.Module, Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[None, torch.Tensor]])
Registers a backward hook on the module.

This function is deprecated in favor of `nn.Module.register_full_backward_hook()` and
the behavior of this function will change in future versions.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_buffer(name: str, tensor: Optional[torch.Tensor], persistent: bool = True)
Adds a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm’s `running_mean`
is not a parameter, but is part of the module’s state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting `persistent` to `False`. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module’s
`state_dict`.

Buffers can be accessed as attributes using given names.


* **Parameters**

    
    * **name** (*string*) – name of the buffer. The buffer can be accessed
    from this module using the given name


    * **tensor** (*Tensor*) – buffer to be registered.


    * **persistent** (*bool*) – whether the buffer is part of this module’s
    `state_dict`.


Example:

```
>>> self.register_buffer('running_mean', torch.zeros(num_features))
```


#### register_forward_hook(hook: Callable[[...], None])
Registers a forward hook on the module.

The hook will be called every time after `forward()` has computed an output.
It should have the following signature:

```
hook(module, input, output) -> None or modified output
```

The input contains only the positional arguments given to the module.
Keyword arguments won’t be passed to the hooks and only to the `forward`.
The hook can modify the output. It can modify the input inplace but
it will not have effect on forward since this is called after
`forward()` is called.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_forward_pre_hook(hook: Callable[[...], None])
Registers a forward pre-hook on the module.

The hook will be called every time before `forward()` is invoked.
It should have the following signature:

```
hook(module, input) -> None or modified input
```

The input contains only the positional arguments given to the module.
Keyword arguments won’t be passed to the hooks and only to the `forward`.
The hook can modify the input. User can either return a tuple or a
single modified value in the hook. We will wrap the value into a tuple
if a single value is returned(unless that value is already a tuple).


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_full_backward_hook(hook: Callable[[torch.nn.modules.module.Module, Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[None, torch.Tensor]])
Registers a backward hook on the module.

The hook will be called every time the gradients with respect to module
inputs are computed. The hook should have the following signature:

```
hook(module, grad_input, grad_output) -> tuple(Tensor) or None
```

The `grad_input` and `grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of `grad_input` in
subsequent computations. `grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in `grad_input` and `grad_output` will be `None` for all non-Tensor
arguments.

**WARNING**: Modifying inputs or outputs inplace is not allowed when using backward hooks and
will raise an error.


* **Returns**

    a handle that can be used to remove the added hook by calling
    `handle.remove()`



* **Return type**

    `torch.utils.hooks.RemovableHandle`



#### register_parameter(name: str, param: Optional[torch.nn.parameter.Parameter])
Adds a parameter to the module.

The parameter can be accessed as an attribute using given name.


* **Parameters**

    
    * **name** (*string*) – name of the parameter. The parameter can be accessed
    from this module using the given name


    * **param** (*Parameter*) – parameter to be added to the module.



#### requires_grad_(requires_grad: bool = True)
Change if autograd should record operations on parameters in this
module.

This method sets the parameters’ `requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).


* **Parameters**

    **requires_grad** (*bool*) – whether autograd should record operations on
    parameters in this module. Default: `True`.



* **Returns**

    self



* **Return type**

    Module



#### share_memory()

#### state_dict(destination=None, prefix='', keep_vars=False)
Returns a dictionary containing a whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.


* **Returns**

    a dictionary containing a whole state of the module



* **Return type**

    dict


Example:

```
>>> module.state_dict().keys()
['bias', 'weight']
```


#### to(\*args, \*\*kwargs)
Moves and/or casts the parameters and buffers.

This can be called as


#### to(device=None, dtype=None, non_blocking=False)

#### to(dtype, non_blocking=False)

#### to(tensor, non_blocking=False)

#### to(memory_format=torch.channels_last)
Its signature is similar to `torch.Tensor.to()`, but only accepts
floating point or complex `dtype\`s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:\`dtype`
(if given). The integral parameters and buffers will be moved
`device`, if that is given, but with dtypes unchanged. When
`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

**NOTE**: This method modifies the module in-place.


* **Parameters**

    
    * **device** (`torch.device`) – the desired device of the parameters
    and buffers in this module


    * **dtype** (`torch.dtype`) – the desired floating point or complex dtype of
    the parameters and buffers in this module


    * **tensor** (*torch.Tensor*) – Tensor whose dtype and device are the desired
    dtype and device for all parameters and buffers in this module


    * **memory_format** (`torch.memory_format`) – the desired memory
    format for 4D parameters and buffers in this module (keyword
    only argument)



* **Returns**

    self



* **Return type**

    Module


Examples:

```
>>> linear = nn.Linear(2, 2)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]])
>>> linear.to(torch.double)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]], dtype=torch.float64)
>>> gpu1 = torch.device("cuda:1")
>>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
>>> cpu = torch.device("cpu")
>>> linear.to(cpu)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16)

>>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
>>> linear.weight
Parameter containing:
tensor([[ 0.3741+0.j,  0.2382+0.j],
        [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
>>> linear(torch.ones(3, 2, dtype=torch.cdouble))
tensor([[0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
```


#### train(mode: bool = True)
Sets the module in training mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. `Dropout`, `BatchNorm`,
etc.


* **Parameters**

    **mode** (*bool*) – whether to set training mode (`True`) or evaluation
    mode (`False`). Default: `True`.



* **Returns**

    self



* **Return type**

    Module



#### training(: bool)

#### type(dst_type: Union[torch.dtype, str])
Casts all parameters and buffers to `dst_type`.


* **Parameters**

    **dst_type** (*type** or **string*) – the desired type



* **Returns**

    self



* **Return type**

    Module



#### xpu(device: Optional[Union[int, torch.device]] = None)
Moves all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.


* **Parameters**

    **device** (*int**, **optional*) – if specified, all parameters will be
    copied to that device



* **Returns**

    self



* **Return type**

    Module



#### zero_grad(set_to_none: bool = False)
Sets gradients of all model parameters to zero. See similar function
under `torch.optim.Optimizer` for more context.


* **Parameters**

    **set_to_none** (*bool*) – instead of setting to zero, set the grads to None.
    See `torch.optim.Optimizer.zero_grad()` for details.


## SubModule Word

@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)>

Copyright (c) 2021 Aymen Jemi SUDO-AI


### class sudoai.models.word.Word2Label(n_class, hidden_size, vocab_size, version='0.1.0', name='word2label', optimizer='sgd', loss='crossentropy', learning_rate=0.01, teacher_forcing_ratio=0.5, momentum=0.0, drop_out=0.1)
Word to Label model.

Useful for task like token classification and profanity detection.


#### optimizer_type()
Code of optimizer like (sgd ~> stochastic gradient descent).


* **Type**

    str



#### loss()
Code of loss function like (nll ~> negative log likelihood loss).


* **Type**

    str



#### max_length()
Maximum word length (see sudoai.utils.MAX_LENGTH).


* **Type**

    int



#### learning_rate()
Learn rate for current model.


* **Type**

    float



#### teacher_forcing_ratio()
Teacher forcing ratio for current model.


* **Type**

    float



#### vocab_size()
Vocab size.


* **Type**

    int



#### name()
Model id.


* **Type**

    str



#### verison()
Model version.


* **Type**

    str



#### momentum()
Value of Optimizer momentum if exist.


* **Type**

    float



#### drop_out()
Value of drop out.


* **Type**

    float



#### hidden_size()
Value of Hidden size of NN.


* **Type**

    int



#### encoder()
Encoder.


* **Type**

    `EncoderRNN`



#### decoder()
Decoder with attention.


* **Type**

    `AttnDecoderRNN`



#### encoder_optimizer()
Optimizer for encoder.


* **Type**

    `torch.optim`



#### decoder_optmizer()
Optimizer for decoder.


* **Type**

    `torch.optim`



#### n_class()
Number of classes.


* **Type**

    int



#### forward(input_tensor: torch.Tensor, target_tensor: Optional[torch.Tensor] = None, do_train: bool = False)
Defines the computation performed at every call.


* **Parameters**

    
    * **input_tensor** (*torch.Tensor*) – Input tensor.


    * **target_tensor** (*torch.Tensor**, **optional*) – Output tensor. Defaults to None.


    * **do_train** (*bool**, **optional*) – If True train mode. Defaults to False.



* **Raises**

    
    * **ValueError** – When input_tensor is not torch.Tensor.


    * **Exception** – When model in train mode and target_tensor is None.



* **Returns**

    In train mode metrics (acc,loss).
    list: Decoded index chars .



* **Return type**

    dict



#### training(: bool)

### class sudoai.models.word.Word2Word(vocab_src: int, vocab_target: int, hidden_size: int, version: str = '0.1.0', name: str = 'word2word', optimizer: str = 'sgd', loss: str = 'crossentropy', learning_rate: float = 0.01, teacher_forcing_ratio: float = 0.5, momentum: float = 0.0, drop_out: float = 0.1)
Word to Word model.

Useful for task like transliteration and translate.


#### optimizer_type()
Code of optimizer like (sgd ~> stochastic gradient descent).


* **Type**

    str



#### loss()
Code of loss function like (nll ~> negative log likelihood loss).


* **Type**

    str



#### max_length()
Maximum word length (see sudoai.utils.MAX_LENGTH).


* **Type**

    int



#### learning_rate()
Learn rate for current model.


* **Type**

    float



#### teacher_forcing_ratio()
Teacher forcing ratio for current model.


* **Type**

    float



#### vocab_src()
Vocab size of source language.


* **Type**

    int



#### vocab_target()
Vocab size of Target language.


* **Type**

    int



#### name()
Model id.


* **Type**

    str



#### verison()
Model version.


* **Type**

    str



#### momentum()
Value of Optimizer momentum if exist.


* **Type**

    float



#### drop_out()
Value of drop out.


* **Type**

    float



#### hidden_size()
Value of Hidden size of NN.


* **Type**

    int



#### encoder()
Encoder.


* **Type**

    `EncoderRNN`



#### decoder()
Decoder with attention.


* **Type**

    `AttnDecoderRNN`



#### encoder_optimizer()
Optimizer for encoder.


* **Type**

    `torch.optim`



#### decoder_optmizer()
Optimizer for decoder.


* **Type**

    `torch.optim`



#### forward(input_tensor: torch.Tensor, target_tensor: Optional[torch.Tensor] = None, do_train: bool = False)
Defines the computation performed at every call.


* **Parameters**

    
    * **input_tensor** (*torch.Tensor*) – Input tensor.


    * **target_tensor** (*torch.Tensor**, **optional*) – Output tensor. Defaults to None.


    * **do_train** (*bool**, **optional*) – If True train mode. Defaults to False.



* **Raises**

    
    * **ValueError** – When input_tensor is not torch.Tensor.


    * **Exception** – When model in train mode and target_tensor is None.



* **Returns**

    In train mode metrics (acc,loss).
    list: Decoded index chars.



* **Return type**

    dict



#### training(: bool)
## SubModule Sequence

@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)>

Copyright (c) 2021 Aymen Jemi SUDO-AI


### class sudoai.models.seq.Seq2Label(n_class, vocab_size, version='0.1.0', name='seq2label', optimizer='sgd', loss='nll', hidden_size=32, learning_rate=0.01, teacher_forcing_ratio=0.5, momentum=0.0, drop_out=0.1)
Sequence to Label model.

Useful for task like sentiment analysis.


#### n_class()
Number of classes.


* **Type**

    int



#### optimizer_type()
Code of optimizer like (sgd ~> stochastic gradient descent).


* **Type**

    str



#### loss()
Code of loss function like (nll ~> negative log likelihood loss).


* **Type**

    str



#### max_length()
Maximum word length (see sudoai.utils.MAX_WORDS).


* **Type**

    int



#### learning_rate()
Learn rate for current model.


* **Type**

    float



#### teacher_forcing_ratio()
Teacher forcing ratio for current model.


* **Type**

    float



#### vocab_size()
Vocab size.


* **Type**

    int



#### name()
Model id.


* **Type**

    str



#### verison()
Model version.


* **Type**

    str



#### momentum()
Value of Optimizer momentum if exist.


* **Type**

    float



#### drop_out()
Value of drop out.


* **Type**

    float



#### hidden_size()
Value of Hidden size of NN.


* **Type**

    int



#### encoder()
Encoder.


* **Type**

    `EncoderRNN`



#### decoder()
Decoder with attention.


* **Type**

    `AttnDecoderRNN`



#### encoder_optimizer()
Optimizer for encoder.


* **Type**

    `torch.optim`



#### decoder_optmizer()
Optimizer for decoder.


* **Type**

    `torch.optim`



#### forward(input_tensor, target_tensor=None, do_train=False)
Defines the computation performed at every call.


* **Parameters**

    
    * **input_tensor** (*torch.Tensor*) – Input tensor.


    * **target_tensor** (*torch.Tensor**, **optional*) – Output tensor. Defaults to None.


    * **do_train** (*bool**, **optional*) – If True train mode. Defaults to False.



* **Raises**

    
    * **ValueError** – When input_tensor is not torch.Tensor.


    * **Exception** – When model in train mode and target_tensor is None.



* **Returns**

    In train mode metrics (acc,loss).
    list: Decoded index words.



* **Return type**

    dict



#### training(: bool)
## SubModule FastText

@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)>

Copyright (c) 2021 Aymen Jemi SUDO-AI


### class sudoai.models.fast.FastModel(id: str, train_path: str, valid_path: Optional[str] = None, version: str = '0.1.0', duration: int = 600, is_ziped: bool = False, auto_metric: Optional[str] = None)
Bases: `object`

Fast text model based on facebook fasttext.


#### train()
Path of train file.


* **Type**

    str



#### valid()
Path of validation file.


* **Type**

    str



#### duration()
Time for autotune.


* **Type**

    int



#### auto_metric_label()
Label for autotune adjust.


* **Type**

    str



#### model()

* **Type**

    `fasttext._FastText`



#### version()
Model version.


* **Type**

    str



#### ziped()
If True current model zip before save.


* **Type**

    bool



#### id()
Model ID.


* **Type**

    str



#### id_trained()
If True current model is trained.


* **Type**

    bool



#### classmethod load(id: str, version: str = '0.1.0', is_ziped=False)
Load saved model with id and version.


* **Parameters**

    
    * **id** (*str*) – Model ID.


    * **version** (*str**, **optional*) – Model version. Defaults to ‘0.1.0’.


    * **is_ziped** (*bool**, **optional*) – If True the model is compressed. Defaults to False.



* **Raises**

    **FileNotFoundError** – When model file not found.



* **Returns**

    FastModel class.



* **Return type**

    `FastModel`



#### predict(input: str, \*\*kwargs)
Predict class from input.


* **Parameters**

    
    * **input** (*str*) – Input text.


    * **clean** (*bool**, **optional*) – If True clean the input text. Defaults to False.


    * **strip_duplicated** (*bool**, **optional*) – If True stripes duplicated chars. Defaults to False.


    * **norm** (*bool**, **optional*) – If True normalize the prediction. Defaults to False.


    * **k** (*int**, **optional*) – Number of classes to predicted. Defaults to 1.


    * **threshold** (*float**, **optional*) – Minimum score for prediction. Defaults to 0.0.



* **Returns**

    Labels predicted.
    Tuple: Labels predicted and scores for each prediction.



* **Return type**

    list



#### quantize(retrain: bool = True, qnorm: bool = True)
Quantize the model reducing the size of the model and it’s memory footprint.


* **Parameters**

    
    * **retrain** (*bool**, **optional*) – Retrain mode. Defaults to True.


    * **qnorm** (*bool**, **optional*) – Normalize current model. Defaults to True.



#### save(\*\*kwargs)
Save current model.


* **Parameters**

    
    * **retrain** (*bool**, **optional*) – In ziped mode if True retrain model.


    * **qnorm** (*bool**, **optional*) – In ziped mode if True Normalize current model.



#### start(\*\*kwargs)
Start Train the model.


* **Parameters**

    
    * **auto** (*bool**, **optional*) – If True model is autotune mode. Defaults to False.


    * **epoch** (*int**, **optional*) – Number of epochs. Defaults to 50.


    * **loss** (*str**, **optional*) – Loss function. Defaults to ‘hs’.


    * **lr** (*float**, **optional*) – Learning rate value. Defaults to None.



* **Raises**

    
    * **FileExistsError** – When train data not found.


    * **FileExistsError** – When validation data not found.


## SubModule Extreme Multi-Label

@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)>

Copyright (c) 2021 Aymen Jemi SUDO-AI


### class sudoai.models.xmltc.HybridXMLTC(n_class: int = 3714, vocab_size: int = 30001, embedding_size: int = 300, hidden_size: int = 256, d_a: int = 256, optimizer: str = 'adam', name: str = 'hybrid_xmltc', momentum: float = 0.0, learning_rate: float = 0.01, multiclass: bool = False, version: str = '0.1.0')
Hybrid Attention for Extreme Multi-Label Text Classification model.

Useful for extreme multi-label classification.


#### n_class()
Number of classes.


* **Type**

    int



#### vocab_size()
Vocab size.


* **Type**

    int



#### embedding_size()
Embedding size.


* **Type**

    int



#### hidden_size()
Value of Hidden size of NN.


* **Type**

    int



#### d_a()
Same as hidden_size.


* **Type**

    int



#### name()
Model id.


* **Type**

    str



#### verison()
Model version.


* **Type**

    str



#### momentum()
Value of Optimizer momentum if exist.


* **Type**

    float



#### learning_rate()
Learn rate for current model.


* **Type**

    float



#### optimizer_type()
Code of optimizer like (sgd ~> stochastic gradient descent).


* **Type**

    str



#### optimizer()
Pytorch Optimizer.


* **Type**

    `torch.optim`



#### xmltc()
Base model.


* **Type**

    `ExtremMutliLabelTextClassification`



#### multiclass()
If True the model is multiclass model.


* **Type**

    bool



#### criterion()
Loss function.


* **Type**

    loss



#### forward(input_tensor, target_tensor=None, do_train=False, threshold=0.5)
Defines the computation performed at every call.


* **Parameters**

    
    * **input_tensor** (*torch.Tensor*) – Input tensor.


    * **target_tensor** (*torch.Tensor**, **optional*) – Output tensor. Defaults to None.


    * **do_train** (*bool**, **optional*) – If True train mode. Defaults to False.


    * **threshold** (*float**, **optional*) – Value of threshold.



* **Raises**

    **ValueError** – When input_tensor is not torch.Tensor.



* **Returns**

    In train mode metrics (acc,loss).



* **Return type**

    dict



#### training(: bool)
