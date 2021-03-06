Welcome to MyCNN by Xu Zhiya. zy-xu16@mails.tsinghua.edu.cn
To start, define a cnn by yourself!
--------------------------------------------------------------------
cnn structure
layers: layers of the cnn (* is required)

      *type:                      type of the layer, could be input layer ('i'), convolutional 
                                  and subsampling layer ('cs'), full connected layer ('fc'), 
                                  and output layer ('o').

      *filterDim:                 dimension of filter, convolutional and
                                  subsampling layer ('cs') only, and real
                                  filter size is filterDim*filterDim*k
                                  where k specifies the numbers of
                                  feature map.

      *numFilters:                numbers of filters, convolutional and
                                  subsampling layer ('cs') only

      *poolDim:                   pool dimension, convolutional and
                                  subsampling layer ('cs') only

      *hiddenUnits                hidden units, full connected layer 
                                  ('fc') only

      activationFunction:         (optional) name of activation function, 
                                  could be 'sigmoid', 'relu' and 'tanh', 
                                  default is 'sigmoid'

      *softmax                    if 1, implement softmax in output
                                  layer, output layer ('o') only
--------------------------------------------------------------------
The input layer, output layer, and at least one convolutional and  subsampling layer are required. In each layer, you can specify an activation function, or use sigmoid in default.

For example

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cnn.layers = {
     struct('type', 'i')
     struct('type', 'cs', 'filterDim', 5, 'numFilters', 6, 'poolDim', 2)
     struct('type', 'o', 'softmax', 0)
};

cnn.layers = {
     struct('type', 'i') %input layer
     struct('type', 'cs', 'filterDim', 5, 'numFilters', 6, 'poolDim', 2, ...
          'activationFunction','relu')
     struct('type', 'fc', 'hiddenUnits', 50, 'activationFunction', 'tanh')
     struct('type', 'o', 'softmax', 1)
};

cnn.layers = {
     struct('type', 'i')
     struct('type', 'cs', 'filterDim', 5, 'numFilters', 6, 'poolDim', 2, ...
            'activationFunction','relu')
     struct('type', 'cs', 'filterDim', 3, 'numFilters', 12,'poolDim', 2, ...
          'activationFunction','relu')
     struct('type', 'fc', 'hiddenUnits', 500, 'activationFunction')
     struct('type', 'fc', 'hiddenUnits', 300, 'activationFunction', 'tanh')
     struct('type', 'o', 'softmax', 1)
};
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

are all valid definations.

Then, load your training data and specify parameter for SGD, and call myCnnTrain to train this cnn.

Finally, load your test data and call myCnnPredict to predict.

A demo is provided in main.m.

Enjoy your time with MyCNN by Xu Zhiya.
zy-xu16@mails.tsinghua.edu.cn
