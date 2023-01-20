# connectionist.layers

## Core layers

One of the core mechanism in connectionist models is the time-averaging input/output. Similar to [Forward Euler method](https://en.wikipedia.org/wiki/Euler_method), the time-averaging input/output is a discrete approximation of the continuous time dynamics.
There are 2 layers in this module that implements this mechanism:

- [TimeAveragedDense][connectionist.layers.TimeAveragedDense] layer: Simulating continuous time with discrete approximation for a single layer input.
- [MultiInputTimeAveraging][connectionist.layers.MultiInputTimeAveraging] layer: Simulating continuous time with discrete approximation for multiple layer inputs.
  
## RNN building blocks

All the RNNs in this module has the time-averaging mechanism.

### Simple RNN

Mainly for the purpose of demonstration, there are 2 simple RNN layers in this module:

- [TimeAveragedRNNCell][connectionist.layers.TimeAveragedRNNCell] layer: Defines one step of compute.
- [TimeAveragedRNN][connectionist.layers.TimeAveragedRNN] layer: Unrolling the `TimeAveragedRNNCell` for multiple steps, similar to `tf.keras.layers.SimpleRNN`, but with time-averaging **output** mechanism.

### PMSP

Building blocks for [PMPS][connectionist.models.PMSP] model:

- [ZeroOutDense][connectionist.layers.ZeroOutDense] layer: A wrapper layer for `tf.keras.layers.Dense` that zero out a portion of the weights in the layer, mainly for model.zero_out "brain" damage API.
- [PMSPCell][connectionist.layers.PMSPCell] layer: Defines one step of compute.
- [PMSPLayer][connectionist.layers.PMSPLayer] layer: Unrolling the `PMSPCell` for multiple steps, describe the entire model architecture of [PMSP][connectionist.models.PMSP].

### Hub-and-spokes

Building blocks for [Hub-and-spokes][connectionist.models.HubAndSpokes] model:

- [HNSSpoke][connectionist.layers.HNSSpoke] layer: Defines one spoke.
- [HNSCell][connectionist.layers.HNSCell] layer: Defines one step of compute.
- [HNSLayer][connectionist.layers.HNSLayer] layer: Unrolling the `HNSCell` for multiple steps, describe the entire model architecture of [HubAndSpokes][connectionist.models.HubAndSpokes].
