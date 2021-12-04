clc
clear all

% first time, so use false
using_existing_bow = false;
%using_existing_bow = true;

if using_existing_bow == false
    fprintf('extracting features...');
    % set the number of centers to use as BOW
    num_centers = 150;
    % set the directory
    files = dir('./images/*.jpg');

    num_images = numel(files);
    features = cell(num_images, 1);
    for i = 1:num_images
        img = rgb2gray(imread(['images/', files(i).name]));
        features{i} = get_features(img);
    end

    % stack vertically
    features_all = cell2mat(features);

    % seed for center's initialization 
    seed = 0;

    my_centers = get_centers(features_all, num_centers, seed);

    h_train = zeros(num_images, num_centers);
    for i = 1:1:num_images
        h_train(i, :) = get_hist(my_centers, features{i});
    end

    save('bow.mat', 'h_train', 'my_centers', 'seed')

elseif using_existing_bow
    fprintf('using existing files!\n');
    bow = load('bow.mat');
    centers = bow.my_centers;
    h_train = bow.h_train;  
end


%% Useful functions
function features = get_features(image)
points = detectSURFFeatures(image);
[features,valid_points] = extractFeatures(image, points, 'Method', 'Surf','FeatureSize' , 64);
features = double(features);

% if too mant images are involved, comment out these plots
figure; 
imshow(image); 
hold on;
plot(valid_points.selectStrongest(10), 'showOrientation', true);
end


function centers = get_centers(feature_all, num_centers, seed)
if nargin == 3
    rng(seed);
end

[total_idx, cluster_center] = kmeans(feature_all, num_centers, 'MaxIter', 200);
centers = cluster_center;
end


function h = get_hist(centers, features)
num_centers = size(centers, 1);
total_num = size(features, 1);
idx = knnsearch(centers, features);
counts = hist(idx, size(centers, 1));
h = counts/sum(counts);
% `h` must be a row vector
assert(isequal(size(h), [1, num_centers]))

% `h` must be normalized
assert((sum(h) - 1)^2 < eps)
end