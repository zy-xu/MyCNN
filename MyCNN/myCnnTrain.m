function cnn = myCnnTrain(cnn, X, y, options, DEBUG)
% CNN implemented by Xu Zhiya
% zy-xu16@mails.tsinghua.edu.cn

% myCnnTrain: Train cnn
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   cnn = myCnnTrain(cnn, X, y, theta)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           cnn         - a cnn whose weights are initialized
%           X           - training data. Should be M*N*D*NUM matrix, where
%                         a single image is of size M*N*D and NUM specifies
%                         numbers of training data
%           y           - training labels
%           options     - options of stochastic gradient descent
%           DEBUG       - (optional) check numerical gradient if DEBUG ==
%                         true, defaults to false
%    Return:
%           cnn         - the updated cnn
%
%   Options (* required)
%       epochs*     - number of epochs through data
%       alpha*      - initial learning rate
%       minibatch*  - size of minibatch
%       momentum    - momentum constant, defualts to 0.9
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

numClasses = max(y);

cnn = myCnnInitParams(cnn, X, numClasses);
theta = myCnnUnrollTheta(cnn);

if ~exist('DEBUG','var')
    DEBUG = false;  % set DEBUG to true to check gradient
end

if DEBUG
    % Using relu may get a error as relu is non-differentiable at 0, ignore
    % the error
    images = X(:,:,:,1:12);
    labels = y(1:12);
    [cost, grad, cnn] = myCnnCost(cnn, images, labels, theta);
    numGrad = computeNumericalGradient( @(t) myCnnCost(cnn, images, labels, t), theta);
    disp([numGrad grad]);
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    assert(diff < 1e-7,...
        'Difference too large. Check your gradient computation again')
    disp('Congratulations! Gradient Check Passed!');
else
    opttheta = minFuncSGD(@(a,b,c,d) myCnnCost(a,b,c,d),cnn,theta,X,y,options);
    cnn = myCnnUpdateTheta(cnn, opttheta);
end







