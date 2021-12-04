clc
clear


path = 'D:/MyDesktop/Cloud LYM/Fall2021/545/Final_Project/processed_dataset_v4/image/*.jpg';

files = dir(path);
clear param 

param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

result = zeros(16189,512);
for c = 1:10
    disp(c)
    file = append(files(c).folder,'\',files(c).name);
    img2 = imread(file);


    
    [gist2, param] = LMgist(img2, '', param);
    
    result(c,:) = gist2;
    save('gist.mat','result')

% 
%     figure
%     subplot(121)
%     imshow(img2)
%     title('Input image')
%     subplot(122)
%     showGist(gist2, param)
%     title('Descriptor')

end




