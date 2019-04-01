data_path = 'data/'
demo_scn_path = fullfile(data_path,'demo'); %CMU+MIT test scenes
test_scn_path = fullfile(data_path,'test_scenes/test_jpg');
test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

load(strcat(train_path_pos,'/svm.mat'),'X','map')

index=randi(length(test_scenes));

delete(fullfile( demo_scn_path, '*.jpg' ));

copyfile(fullfile( test_scn_path, test_scenes(i).name ), demo_scn_path);


[bboxes, confidences, image_ids] = run_detector(demo_scn_path, w, b, feature_params);



visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path)

% visualize_detections_by_confidence(bboxes, confidences, image_ids, test_scn_path, label_path);

% performance to aim for
% random (stater code) 0.001 AP
% single scale ~ 0.2 to 0.4 AP
% multiscale, 6 pixel step ~ 0.83 AP
% multiscale, 4 pixel step ~ 0.89 AP
% multiscale, 3 pixel step ~ 0.92 AP
