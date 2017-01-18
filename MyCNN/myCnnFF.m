function cnn = myCnnFF(cnn,X)
% CNN implemented by Xu Zhiya
% zy-xu16@mails.tsinghua.edu.cn

% myCnnFF: Implentet Feedforward
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   cnn = myCnnFF(cnn,X)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           cnn         - a cnn whose weights are initialized or specified
%           X           - training data. Should be M*N*D*NUM matrix, where
%                         a single image is of size M*N*D and NUM specifies
%                         numbers of training data
%    Return:
%           cnn         - the updated cnn
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


numImages = size(X,4);
numLayers = size(cnn.layers,1);

for i = 1:numLayers
    if strcmp(cnn.layers{i}.type, 'i')
        cnn.layers{i}.activations = X;
    elseif strcmp(cnn.layers{i}.type, 'cs')
        filterDim = cnn.layers{i}.filterDim;
        featureDim = cnn.layers{i-1}.outDim(1);
        convDim = featureDim - filterDim + 1;
        numFilters = cnn.layers{i}.numFilters;
        cnn.layers{i}.convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);
        
        % convolution
        for imageNum = 1:numImages
            for filterNum = 1:numFilters
                curFilter = cnn.layers{i-1}.W(:,:,:,filterNum);
                
                curFilter = rot90(curFilter,2);
                im = cnn.layers{i-1}.activations(:, :, :,imageNum);
                convolvedFeatures = convn(im,curFilter,'valid');
                convolvedFeatures = convolvedFeatures + cnn.layers{i-1}.b(filterNum);
                convolvedFeatures = cnn.layers{i-1}.realActivationFunction(convolvedFeatures);
                cnn.layers{i}.convolvedFeatures(:, :, filterNum, imageNum) = convolvedFeatures;
            end
        end
        
        %pooling
        cnn.layers{i}.activations = zeros([cnn.layers{i}.outDim numImages]);
        poolDim = cnn.layers{i}.poolDim;
        mask = ones(poolDim,poolDim)/poolDim^2;
        for imageNum = 1:numImages
            for filterNum = 1:numFilters
                tmp = conv2(cnn.layers{i}.convolvedFeatures(:,:,filterNum,imageNum),mask,'valid');
                cnn.layers{i}.activations(:,:,filterNum,imageNum) = tmp(1:poolDim:end,1:poolDim:end);
            end
        end
        
    elseif strcmp(cnn.layers{i}.type, 'fc') ||  (strcmp(cnn.layers{i}.type, 'o') && cnn.layers{i}.softmax == false)
        activations = reshape(cnn.layers{i-1}.activations,[],numImages);
        cnn.layers{i}.activations = cnn.layers{i-1}.realActivationFunction(cnn.layers{i-1}.W * activations + repmat(cnn.layers{i-1}.b,1,numImages));
    elseif strcmp(cnn.layers{i}.type, 'o') && cnn.layers{i}.softmax == true
        % Softmax with sigmoid
        activations = reshape(cnn.layers{i-1}.activations,[],numImages);
        activations = cnn.layers{i-1}.W * activations + repmat(cnn.layers{i-1}.b,1,numImages);
        activations = bsxfun(@minus,activations,max(activations,[],1));
        activations = exp(activations);
        cnn.layers{i}.activations = bsxfun(@rdivide, activations, sum(activations));
    end
end





