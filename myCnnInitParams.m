function cnn = myCnnInitParams(cnn, X, numClasses)
% CNN implemented by Xu Zhiya
% zy-xu16@mails.tsinghua.edu.cn

% myCnnInitParams: Initialize weights of CNN
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   cnn = myCnnInitParams(cnn, X, numClasses)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           cnn         - a cnn whose structure is defined
%           X           - training data. Should be M*N*D*NUM matrix, where
%                         a single image is of size M*N*D and NUM specifies
%                         numbers of training data
%           numClasses  - classes of training data
%    Return:
%           cnn         - the cnn whose weights are initialized
%    ---------------------------------------------------------------------------------
%
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


numLayers = size(cnn.layers,1);
cnn.layers{numLayers}.hiddenUnits = numClasses;

% Specify each layer's activation function 
for i = 2:numLayers
    if ~isfield(cnn.layers{i},'activationFunction')
        cnn.layers{i}.activationFunction = 'sigmoid';
    end
    
    if (i == numLayers) && (cnn.layers{i}.softmax) == 1 && ...
            (~strcmp(cnn.layers{i}.activationFunction, 'sigmoid'))
        warning('Will aply sigmoid function to softmax layer');
        cnn.layers{i}.activationFunction = 'sigmoid';
    end
    
    switch cnn.layers{i}.activationFunction
        case 'sigmoid'
            cnn.layers{i-1}.realActivationFunction = @(x)sigmoid(x);
            cnn.layers{i-1}.realGradientFunction = @(x)sigmoidGradient(x);
        case 'tanh'
            cnn.layers{i-1}.realActivationFunction = @(x)tanh(x);
            cnn.layers{i-1}.realGradientFunction = @(x)tanhGradient(x);
        case 'relu'
            cnn.layers{i-1}.realActivationFunction = @(x)relu(x);
            cnn.layers{i-1}.realGradientFunction = @(x)reluGradient(x);
    end
end

% Initialize weights
for i = 1:numLayers
    if strcmp(cnn.layers{i}.type, 'i')
        outDim = size(X);
        cnn.layers{i}.outDim = outDim(1:3);
    elseif strcmp(cnn.layers{i}.type, 'cs')
        filterDim = cnn.layers{i}.filterDim;
        numFilters = cnn.layers{i}.numFilters;
        filterDim3 = cnn.layers{i-1}.outDim(3);
        Wc = 1e-1 * randn(filterDim,filterDim,filterDim3,numFilters);
        cnn.layers{i-1}.W = Wc;
        cnn.layers{i-1}.b = zeros(numFilters,1);
        
        featureDim = cnn.layers{i-1}.outDim(1);      
        poolDim = cnn.layers{i}.poolDim;
        pooledSize = featureDim - filterDim + 1; % dimension of convolved image
        pooledSize = pooledSize/poolDim;
        cnn.layers{i}.outDim = [pooledSize pooledSize numFilters];
    elseif strcmp(cnn.layers{i}.type, 'fc') ||  strcmp(cnn.layers{i}.type, 'o')
        if strcmp(cnn.layers{i-1}.type, 'cs') || strcmp(cnn.layers{i-1}.type, 'i')
            hiddenSize = cnn.layers{i-1}.outDim(1).^2 * cnn.layers{i-1}.outDim(3);
            hiddenUnits = cnn.layers{i}.hiddenUnits;
            r  = sqrt(6) / sqrt(hiddenUnits + hiddenSize + 1);
            Wf = rand(hiddenUnits, hiddenSize) * 2 * r - r;
            cnn.layers{i-1}.W = Wf;
            cnn.layers{i-1}.b = zeros(hiddenUnits,1);
            cnn.layers{i}.outDim = hiddenUnits;
        else
            hiddenSize = cnn.layers{i-1}.outDim(1);
            hiddenUnits = cnn.layers{i}.hiddenUnits;
            r  = sqrt(6) / sqrt(hiddenUnits + hiddenSize + 1);
            Wf = rand(hiddenUnits, hiddenSize) * 2 * r - r;
            cnn.layers{i-1}.W = Wf;
            cnn.layers{i-1}.b = zeros(hiddenUnits,1);
            cnn.layers{i}.outDim = hiddenUnits;
        end
    end
end

kk = 1;