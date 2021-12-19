%#------  Test trained network ------#%

% Function to test network
function test(net, image)

% Read Image
Img = imread(image);

% Resize image
Resize = imresize(Img, [224, 224]);

% Transfer image to network
[Label, Prob] = classify(net,Resize);

% Now show image, label & probability 
figure;
imshow(Resize);
title({char(Label), num2str(max(Prob),2)});
end

% Use " test(net, image1) " to see result