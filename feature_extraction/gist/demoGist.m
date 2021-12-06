clc
clear


path = 'E:\EECS545\processed_dataset_v4\image\*.jpg';

files = dir(path);
clear param 

param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

result = zeros(16188,512);
for c = 1:1:length(files)
    disp(c)
    file = [files(c).folder,'\',files(c).name];
    img2 = imread(file);
    img2 = imresize(img2, [500, 500]);


    
    [gist2, param] = LMgist(img2, '', param);
    
    result(c,:) = gist2;
    % save('gist.mat','result')
% save('gist.mat','result')

% 
%     figure
%     subplot(121)
%     imshow(img2)
%     title('Input image')
%     subplot(122)
%     showGist(gist2, param)
%     title('Descriptor')

end
save('gist.mat','result')



