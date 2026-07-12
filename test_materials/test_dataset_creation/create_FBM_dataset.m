%% create_FBM_dataset.m
% Генерация набора 2D FBM изображений методом Midpoint Displacement
%
% Для каждого H = 0:0.1:1 создаются две реализации.
% Сохраняются:
%   1) исходные данные im в формате .mat
%   2) изображение для просмотра в формате .png


clc;
clear;


%% Параметры генерации

% Размер изображения:
% N = 2^MaxLevel + 1
MaxLevel = 8;       % 257 x 257


% Значения параметра Херста
H_values = 0:0.1:1;


% Две разные реализации для каждого H
seeds = [19560709, 123456];


%% Папки для сохранения

mat_folder = "FBM_mat";
png_folder = "FBM_png";


% Создание папок при необходимости

if ~exist(mat_folder, 'dir')
    mkdir(mat_folder);
end

if ~exist(png_folder, 'dir')
    mkdir(png_folder);
end


%% Генерация изображений

for i = 1:length(H_values)

    H = H_values(i);


    for j = 1:length(seeds)

        seed = seeds(j);


        %% Генерация FBM поверхности

        im = midpoint2D(MaxLevel, H, seed);


        %% Формирование имени файла

        base_name = sprintf( ...
            "FBM_H%.1f_seed%d", ...
            H, seed);


        %% Сохранение исходной матрицы MATLAB

        mat_filename = fullfile( ...
            mat_folder, ...
            base_name + ".mat");


        save(mat_filename, "im", "H", "seed");


        %% Сохранение изображения PNG

        png_filename = fullfile( ...
            png_folder, ...
            base_name + ".png");


        % Для визуализации переводим значения в [0,1]
        im_png = mat2gray(im);


        imwrite(im_png, png_filename);


        %% Информация о прогрессе

        fprintf( ...
            "Saved H=%.1f, seed=%d\n", ...
            H, seed);


    end

end


disp("FBM dataset generation finished!");