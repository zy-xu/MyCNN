function preds = myCnnPredict(cnn,X)
% CNN implemented by Xu Zhiya
% zy-xu16@mails.tsinghua.edu.cn

% myCnnPredict: Predict labels of test set
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   preds = myCnnPredict(cnn,X)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           cnn         - a cnn whose weights are initialized or specified
%           X           - test data. Should be M*N*D*NUM matrix, where
%                         a single image is of size M*N*D and NUM specifies
%                         numbers of test data
%    Return:
%           preds       - predictions of test set
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

cnn = myCnnFF(cnn,X);
[~,preds] = max(cnn.layers{end}.activations,[],1);
preds = preds';
