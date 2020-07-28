clear 
close all

%% Parameter setting
%tau = [3 0.05];
min_depth = 1e-3;
max_depth = 80;
num_samples = 141;
abs_rel = zeros(1,num_samples);
sq_rel  = zeros(1,num_samples);
rmse    = zeros(1,num_samples);
rmse_log = zeros(1,num_samples);
a1  = zeros(1,num_samples);
a2  = zeros(1,num_samples);
a3  = zeros(1,num_samples);
err_2 = zeros(1,num_samples);
err_3 = zeros(1,num_samples);
err_5 = zeros(1,num_samples);
density = zeros(1,num_samples);

[raw,stereo] = textread('matches.txt', '%s %s');
for i = 1:141  
    %% read lidar points
    [~,name_stereo,~] = fileparts(stereo{i});
    name = [name_stereo '.png'];
    stereo_disp_name_gt  = ['Kitti_141/disp_noc_0/' name];
    stereo_disp_name_est = ['best_model_images/' name];
    
    stereo_disp_gt  = double(imread(stereo_disp_name_gt))/256;
    stereo_disp_est = double(imread(stereo_disp_name_est))/256;
    
    [h,w] = size(stereo_disp_gt);
    mask1 = stereo_disp_est>0 & stereo_disp_gt>0;
    mask2 = stereo_disp_gt>0;
    density(i) = sum(mask1(:))/sum(mask2(:));
       
    err_3(i) = disp_error (stereo_disp_gt,stereo_disp_est,[3 0.05]);  
    err_2(i) = disp_error (stereo_disp_gt,stereo_disp_est,[2 0.05]);  
    err_5(i) = disp_error (stereo_disp_gt,stereo_disp_est,[5 0.05]);  

    %save log disparity, adding text in image
    position = [0,300];
    text_str = [num2str(err_3(i)*100,'%0.2f') '%'];
    log_disp_est = logarithm(stereo_disp_est, 0.9);
    log_disp_est = insertText(log_disp_est, position, text_str,...
    'FontSize',48,'BoxOpacity',0,'TextColor', [100, 62,28]);

    disp_path = './display_disp';
    if ~exist(disp_path)
        mkdir(disp_path)
    end  
    name_stereo = [name(1:9) '_log.png'];
    imwrite(log_disp_est, [disp_path '/' name_stereo]);

    % turn disparity to depth
    imgSize = size(stereo_disp_gt);
    if imgSize[1] == 1242;
        focal = 721.5377;
    elseif imgSize[1] == 1241;
        focal = 718.856;        
    elseif imgSize[1] == 1224;
        focal = 707.0493;        
    elseif imgSize[1] == 1238;
        focal = 718.3351;
    elseif imgSize[1] == 1226;
        focal = 707.0912;
    end      
    baseline = 0.54;
    mask = stereo_disp_gt>0;
    stereo_depth = depth2disparity(stereo_disp_gt,baseline,focal);
    stereo_depth_est = depth2disparity(stereo_disp_est,baseline,focal);    
    stereo_depth_est(stereo_depth_est < min_depth) = min_depth;
    stereo_depth_est(stereo_depth_est > max_depth) = max_depth;
    [abs_rel(i), sq_rel(i), rmse(i), rmse_log(i), a1(i), a2(i), a3(i)] = depth_error(stereo_depth(mask),stereo_depth_est(mask));
    fprintf('%s results: abs_rel %.4f, bad_2 %.4f, bad_3 %.4f, bad_5 %.4f, a1 %.4f \n',...
            name_stereo, abs_rel(i),   err_2(i),   err_3(i),   err_5(i),   a1(i))
    
end

abs_rel_mean = mean(abs_rel);
err2_mean = mean(err_2); err3_mean = mean(err_3); err5_mean = mean(err_5);
a1_mean = mean(a1); 
fprintf('mean results: \n abs_rel %.4f, bad_2 %.4f, bad_3 %.4f, bad_5 %.4f, a1 %.4f\n',...
                     abs_rel_mean, err2_mean,  err3_mean,  err5_mean,  a1_mean)

function img_log = logarithm(img,c)
img_log = c*log(double(img)+1);
img_log = im2uint8(mat2gray(img_log));
end
