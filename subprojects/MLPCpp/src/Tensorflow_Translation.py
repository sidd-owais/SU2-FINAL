#!/usr/bin/env python

## \file Tensorflow_Translation.py
#  \brief Python script for translating a trained Tensorflow model 
#         to a file compatible with the MLPCpp library classes.
#  \author E.C.Bunschoten
#
# MLPCpp Project Website: https://github.com/EvertBunschoten/MLPCpp
#
# Copyright (c) 2023 Evert Bunschoten

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



def Translate_Tensorflow_MLP(file_out, input_names, output_names, model, input_min=[], input_max=[], output_min=[], output_max=[]):
    # This function writes the MLP to a format which can be read by the SU2 MLP import tool
    # Inputs:
    # - file_out: output file name without extension
    # - input_names: list of strings with the variable names of the MLP input(s)
    # - output names: list of strings with the variable names of the MLP output(s)
    # - model: tensorflow.keras.model; the trained model
    # - input_min: lower normalization values for the input
    # - input_max: upper normalization values for the input
    # - output_min: lower normalization values for the output
    # - output_max: upper normalization values for the output

    # MLP config
    model_config = model.get_config()

    # Number of input variables in the model
    n_inputs = model_config['layers'][0]['config']['batch_input_shape'][1]
    # Number of output variables in the model
    n_outputs = model_config['layers'][-1]['config']['units']

    # Checking if number of provided input and output names are equal to those in the model
    if not n_inputs == len(input_names):
        raise Exception("Number of provided input names unequal to the number of inputs in the model")
    if not n_outputs == len(output_names):
        raise Exception("Number of provided output names unequal to the number of outputs in the model")

    if len(input_max) != len(input_min):
        raise Exception("Upper and lower input normalizations should have the same length")
    if len(output_max) != len(output_min):
        raise Exception("Upper and lower output normalizations should have the same length")

    if len(input_max) > 0 and len(input_min) != n_inputs:
        raise Exception("Input normalization not provided for all inputs")

    if len(output_max) > 0 and len(output_min) != n_outputs:
        raise Exception("Output normalization not provided for all outputs")


    # Creating output file
    fid = open(file_out+'.mlp', 'w+')
    fid.write("<header>\n\n")
    n_layers = len(model_config['layers'])

    # Writing number of neurons per layer
    fid.write('[number of layers]\n%i\n\n' % n_layers)
    fid.write('[neurons per layer]\n')
    activation_functions = []

    for iLayer in range(n_layers-1):
        layer_class = model_config['layers'][iLayer]['class_name']
        if layer_class == 'InputLayer':
            # In case of the input layer, the input shape is written instead of the number of units
            activation_functions.append('linear')
            n_neurons = model_config['layers'][iLayer]['config']['batch_input_shape'][1]
        else:
            activation_functions.append(model_config['layers'][iLayer]['config']['activation'])
            n_neurons = model_config['layers'][iLayer]['config']['units']

        fid.write('%i\n' % n_neurons)
    fid.write('%i\n' % n_outputs)

    activation_functions.append('linear')

    # Writing the activation function for each layer
    fid.write('\n[activation function]\n')
    for iLayer in range(n_layers):
        fid.write(activation_functions[iLayer] + '\n')

    # Writing the input and output names
    fid.write('\n[input names]\n')
    for input in input_names:
        fid.write(input + '\n')
    
    if len(input_min) > 0:
        fid.write('\n[input normalization]\n')
        for i in range(len(input_names)):
            fid.write('%+.16e\t%+.16e\n' % (input_min[i], input_max[i]))
    
    fid.write('\n[output names]\n')
    for output in output_names:
        fid.write(output+'\n')
    
    if len(output_min) > 0:
        fid.write('\n[output normalization]\n')
        for i in range(len(output_names)):
            fid.write('%+.16e\t%+.16e\n' % (output_min[i], output_max[i]))

    fid.write("\n</header>\n")
    # Writing the weights of each layer
    fid.write('\n[weights per layer]\n')
    for layer in model.layers:
        fid.write('<layer>\n')
        weights = layer.get_weights()[0]
        for row in weights:
            fid.write("\t".join(f'{w:+.16e}' for w in row) + "\n")
        fid.write('</layer>\n')
    
    # Writing the biases of each layer
    fid.write('\n[biases per layer]\n')
    
    # Input layer biases are set to zero
    fid.write('%+.16e\t%+.16e\t%+.16e\n' % (0.0, 0.0, 0.0))

    for layer in model.layers:
        biases = layer.get_weights()[1]
        fid.write("\t".join([f'{b:+.16e}' for b in biases]) + "\n")

    fid.close()