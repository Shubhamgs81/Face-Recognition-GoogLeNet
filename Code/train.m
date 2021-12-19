%#------ Face Recognition using GoogLeNet ------#%

%#------ Training Model ------#%

%#------1.  Load Dataset------#%

% Create & label Dataset
Dataset = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Define Training & Validation Dataset
Training_Dataset=Dataset;
Validation_Dataset=Dataset;


%#------2.  Load Network------#%

% Set Network
net = googlenet;
analyzeNetwork(net)

% Store Input size of layer 1
Input_Layer_Size = net.Layers(1).InputSize;

% Store whole architecture of network(Googlenet)
Layer_Graph = layerGraph(net);



%#------3.  Modify Required Layers ------#%

% Store layer 142 & 144
Feature_Learner = net.Layers(142);
Output_Classifier = net.Layers(144);

% Store number of classes in Dataset
Number_of_Classes = numel(categories(Training_Dataset.Labels));

% Create new fullyConnectedLayer(142) & classificationLayer(144)
New_Feature_Learner = fullyConnectedLayer(Number_of_Classes, ...
    'Name', 'Facial Feature Learner', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
New_Classifier_Layer = classificationLayer('Name', 'Face Classifier');

% Replace layer 142 & 144
Layer_Graph = replaceLayer(Layer_Graph, Feature_Learner.Name, New_Feature_Learner);
Layer_Graph = replaceLayer(Layer_Graph, Output_Classifier.Name, New_Classifier_Layer);
analyzeNetwork(Layer_Graph)



%#------4.  Modify Images ------#%

% Define range of modification
Pixel_Range = [-30 30];
Scale_Range = [0.9 1.1];

% Now modify images
Image_Augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandXTranslation', Pixel_Range, ...
    'RandYTranslation', Pixel_Range,... 
     'RandXScale', Scale_Range, ...
     'RandYScale', Scale_Range);

 % Resize image for layer 1 of Googlenet
Augmented_Training_Image = augmentedImageDatastore(Input_Layer_Size(1:2), Training_Dataset, ...
    'DataAugmentation', Image_Augmenter);

Augmented_Validation_Image = augmentedImageDatastore(Input_Layer_Size(1:2),Validation_Dataset);



%#------5.  Train Network ------#%

% Specify training Option
Size_of_Minibatch = 5;
Validation_Frequency = floor(numel(Augmented_Training_Image.Files)/Size_of_Minibatch);
Training_Options = trainingOptions('sgdm',...
    'MiniBatchSize', Size_of_Minibatch, ...
    'MaxEpochs', 10,...
    'InitialLearnRate', 3e-4,...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Augmented_Validation_Image, ...
    'ValidationFrequency', Validation_Frequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Start training
net = trainNetwork(Augmented_Training_Image, Layer_Graph, Training_Options);




