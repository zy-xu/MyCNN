function [cost, grad, cnn] = myCnnCost(cnn, X, y, theta)
% CNN implemented by Xu Zhiya
% zy-xu16@mails.tsinghua.edu.cn

% myCnnCost: Implement backpropagation
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   [cost, grad, cnn] = myCnnCost(cnn, X, y, theta)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           cnn         - a cnn whose weights are initialized or specified
%           X           - training data. Should be M*N*D*NUM matrix, where
%                         a single image is of size M*N*D and NUM specifies
%                         numbers of training data
%           y           - training labels
%    Return:
%           cost        - cost of a single iteration
%           grad        - gradients of a single iteration, unroll to one
%                         column
%           cnn         - the updated cnn
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

numImages = size(X,4);
% cnn.layers{1}.activations = X;
% cnn.layers{1}.outDim = size(X);
% cnn = myCnnFF(cnn,X);
numLayers = size(cnn.layers,1);
numClasses = cnn.layers{numLayers}.hiddenUnits;

if exist('theta','var')
    cnn = myCnnUpdateTheta(cnn,theta);
end

cnn = myCnnFF(cnn,X);

% Compute cost
if cnn.layers{numLayers}.softmax == true
    groundTruth = full(sparse(y, 1:numImages, 1, numClasses, numImages));
    cost = -1./numImages*groundTruth(:)'*log(cnn.layers{end}.activations(:));
    cnn.layers{numLayers}.delta = -(groundTruth - cnn.layers{end}.activations);
else
    groundTruth = full(sparse(y, 1:numImages, 1, numClasses, numImages));
    if strcmp(cnn.layers{numLayers-1}.activationFunction,'tanh')
        groundTruth(groundTruth == 0) = -1;
    end
%     a_nl = cnn.layers{end}.activations;
%     cost = sum(sum(-groundTruth .* log(a_nl) - (1-groundTruth) .* log(1 - a_nl))) / numImages;
%     cnn.layers{numLayers}.delta = a_nl - groundTruth;
    a_nl = cnn.layers{end}.activations;
    cost = sum(sum((a_nl - groundTruth).^2) / 2) /numImages;
    cnn.layers{numLayers}.delta = -(groundTruth - a_nl) .* cnn.layers{numLayers-1}.realGradientFunction(a_nl) ;
end

% Compute delta
for i = numLayers-1:-1:2
    % Delta of 'cs'
    if strcmp(cnn.layers{i}.type, 'cs')
        if strcmp(cnn.layers{i+1}.type, 'fc') || strcmp(cnn.layers{i+1}.type, 'o')
             % Previous layer is 'fc' or 'o'
            delta_s = cnn.layers{i}.W' * cnn.layers{i+1}.delta;
            delta_s = reshape(delta_s,[cnn.layers{i}.outDim numImages]);
        else
            % Previous layer is 'cs'
            numFilters = cnn.layers{i+1}.numFilters;
            delta_s = zeros([cnn.layers{i}.outDim numImages]);
            for imageNum = 1:numImages
                delta = zeros(cnn.layers{i}.outDim(1:3));
                for filterNum = 1:numFilters
                    delta = delta + convn(cnn.layers{i+1}.delta(:,:,filterNum,imageNum),...
                        cnn.layers{i}.W(:,:,:,filterNum),'full');
                end
                delta_s(:,:,:,imageNum) = delta;
            end
            % !!!real convolution, dim3 should also be fliped
            delta_s = flip(delta_s,3);
        end
        
        convDim = cnn.layers{i}.outDim(1) * cnn.layers{i}.poolDim;
        numFilters = cnn.layers{i}.numFilters;
        delta_c = zeros(convDim,convDim,numFilters,numImages);
        for imageNum = 1:numImages
            for filterNum = 1:numFilters
                delta_c(:,:,filterNum,imageNum) = (1./cnn.layers{i}.poolDim^2) * ...
                    kron((delta_s(:,:,filterNum,imageNum)), ones(cnn.layers{i}.poolDim));
            end
        end
        cnn.layers{i}.delta = delta_c .* cnn.layers{i-1}.realGradientFunction(cnn.layers{i}.convolvedFeatures);
    elseif strcmp(cnn.layers{i}.type, 'fc')
        % Delta of 'fc'
        cnn.layers{i}.delta = cnn.layers{i}.W' * cnn.layers{i+1}.delta ...
            .* cnn.layers{i-1}.realGradientFunction(cnn.layers{i}.activations);
    end
end

% Compute gradient
for i = numLayers-1:-1:1
    if strcmp(cnn.layers{i+1}.type, 'cs')
        numFilters = cnn.layers{i+1}.numFilters;
        Wgrad = zeros(size(cnn.layers{i}.W));
        bgrad = zeros(size(cnn.layers{i}.b));
        filterDim = cnn.layers{i+1}.filterDim;
        if strcmp(cnn.layers{i}.type, 'i')
            filterDim3 = size(X,3);
        else
            filterDim3 = cnn.layers{i}.outDim(3);
        end
        for filterNum = 1:numFilters
            Wgrad_i = zeros(filterDim,filterDim,filterDim3);
            for imageNum = 1:numImages
                Wgrad_i = Wgrad_i + convn(cnn.layers{i}.activations(:,:,:,imageNum), ...
                    rot90(cnn.layers{i+1}.delta(:,:,filterNum,imageNum),2), 'valid');
            end
            Wgrad(:,:,:,filterNum) = flip(Wgrad_i,3)/numImages;
            b_i = cnn.layers{i+1}.delta(:,:,filterNum,:);
            bgrad(filterNum) = sum(b_i(:)) / numImages;
        end
        cnn.layers{i}.Wgrad = Wgrad;
        cnn.layers{i}.bgrad = bgrad;
        
    elseif strcmp(cnn.layers{i+1}.type, 'fc') || strcmp(cnn.layers{i+1}.type, 'o')
        if strcmp(cnn.layers{i}.type, 'cs')
            activations = reshape(cnn.layers{i}.activations,[],numImages);
            cnn.layers{i}.Wgrad = cnn.layers{i+1}.delta * activations' / numImages;
        else
            cnn.layers{i}.Wgrad = cnn.layers{i+1}.delta * cnn.layers{i}.activations' / numImages;
        end
        cnn.layers{i}.bgrad = sum(cnn.layers{i+1}.delta,2) / numImages;
    end
end

% Unroll gradient
grad = [];
numLayers = size(cnn.layers,1);
for i = 1:numLayers-1
    Wgrad = cnn.layers{i}.Wgrad(:);
    grad = [grad;Wgrad];
end

for i = 1:numLayers-1
    bgrad = cnn.layers{i}.bgrad(:);
    grad = [grad;bgrad];
end

kk = 1;






