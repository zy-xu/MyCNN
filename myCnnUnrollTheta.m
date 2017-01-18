function theta = myCnnUnrollTheta(cnn)
% CNN implemented by Xu Zhiya
% zy-xu16@mails.tsinghua.edu.cn

% myCnnUnrollTheta: Unroll the weights of to one column
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   theta = myCnnUnrollTheta(cnn)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           cnn         - a cnn whose weights are initialized or specified
%    Return:
%           theta       - unrolled weights
%    ---------------------------------------------------------------------------------
% cnn structure
%   layers: layers of the cnn
%       type:                       type of the layer, could be input layer ('i'), convolutional 
%                                   and subsampling layer ('cs'), full connected layer ('fc'), 
%                                   and output layer ('o').
%
%       filterDim:                  dimension of filter, convolutional and
%                                   subsampling layer ('cs') only, and real
%                                   filter size is filterDim*filterDim*k
%                                   where k specifies the numbers of
%                                   feature map.
%
%       numFilters:                 numbers of filters, convolutional and
%                                   subsampling layer ('cs') only
%
%       poolDim:                    pool dimension, convolutional and
%                                   subsampling layer ('cs') only
%
%       hiddenUnits                 hidden units, full connected layer 
%                                   ('fc') and output layer ('o') only
%
%       activationFunction:         name of activation function, could be
%                                   'sigmoid', 'relu' and 'tanh', default
%                                   is 'sigmoid'
%
%       realActivationFunction:     function handle of activation function
%
%       realGradientFunction:       function handle of the gradients of the 
%                                   activation function
%
%       outDim:                     output dimension
%
%       W:                          weights
%
%       b:                          bias
%
%       convolvedFeatures:          convolved features
%
%       activations:                'input' of the next layer
%
%       delta:                      sensitivities
%
%       Wgrad:                      gradients of weights
%
%       bgrad:                      gradients of bias
%
%       softmax                     if 1, implement softmax in output
%                                   layer, output layer ('o') only

theta = [];
numLayers = size(cnn.layers,1);
for i = 1:numLayers-1
    W = cnn.layers{i}.W(:);
    theta = [theta;W];
end

for i = 1:numLayers-1
    b = cnn.layers{i}.b(:);
    theta = [theta;b];
end