clear all; clc; format compact;

%set(0,'DefaultFigureWindowStyle','docked') %figures in tab

%% Path settings for script
% Dependencies copied into shared dropbox folder 2019/12/23: add current path and set paths in relation to it
mfile_full = matlab.desktop.editor.getActiveFilename;
[dir_mfile, mfile] = fileparts(mfile_full);
addpath(dir_mfile);

cell_dir = split(dir_mfile, filesep);
dir_clin = fullfile(cell_dir{1:end-1}, filesep, 'DataClinical');
dir_spec = fullfile(cell_dir{1:end-1}, filesep, 'Spectrograms');
dir_code = fullfile(cell_dir{1:end-1}, filesep, 'CodeEyal');
dir_eeg = fullfile(cell_dir{1:end-1}, filesep, 'DataEEG');
dir_features = fullfile(cell_dir{1:end-1}, filesep, 'DataFeatures');

%% Pull up clinical data
% close all;
% fclose all;
tbl_clin = EEG_AMS_LoadClinicalData(dir_clin);
clindata=tbl_clin([1:28,30:46,48:69,71:73,75,77:114,116:175,177:192],[4,13]); % subject id + time of measurement

% Pull up clinical data (CAMS score)
% clindata=tbl_clin([1:28,30:46,48:69,71:73,75,77:114,116:175,177:192],[4 598]); %subject id + cam severity score (is this the right one?)
% datatable_clin=tbl_clin([1:28,30:46,48:69,71:73,75,77:114,116:175,177:192],[4, 99:102,107,598]); % subject ID + raw short/long + corrected short/long + 3D-CAMS + CAM-S severity calculated?

%% Load EEG data from individual file
% dirname = 'D:\Dropbox (Partners HealthCare)\EEG_CAMS\Delrium_NonICU_GretaMarinka\DataEEG';
% dirname = 'D:\Dropbox (Partners HealthCare)\EEG_CAMS\DeliriumSeverity_EEG_NonIntubated_Eyal\DataEEG';
cd(dir_eeg);

% Find files
files = dir('DELIRIUM_NON_ICU_AMSD*.mat');

% Freq parameters for EEG filter
freq_min=0.5;
freq_max=30;
flag_notch = true;
s=1; %counter

% FigDocked;
% clf;
% tic;

for i_file = 1:numel(files)
    filename = files(i_file).name;
    % Load data
    load(filename, 'Fs', 'channels', 'data');               % Fs, channels, data, start_time
    ts = (0:size(data, 2)-1)/Fs;                            % Convert from samples to timestamps based on Fs (s)
    channels = EEG_CAMS_ChannelsToCell(channels);           % Standardize channels format, some saved as char and some as nested cell
    
    % decide the CAM test time
    CAM_time = table2array(clindata(i_file,2)); CAM_time=num2str(CAM_time);
    if length(CAM_time) == 4
    CAM_time_hour=str2num(CAM_time(1:2))*3600;               % translated to seconds
    CAM_time_min=str2num(CAM_time(3:4))*60;                  % translated to seconds
    CAM_time=CAM_time_hour+CAM_time_min;                     % in [sec]
    else
    CAM_time_hour=str2num(CAM_time(1))*3600;                 % translated to seconds
    CAM_time_min=str2num(CAM_time(2:3))*60;                  % translated to seconds
    CAM_time=CAM_time_hour+CAM_time_min;                     % in [sec]
    end
    
    % decide the starting time of EEG recording
    EEG_start_time_hour=str2num(filename(35:36))*3600;       % translated to seconds
    EEG_start_time_min=str2num(filename(37:38))*60;          % translated to seconds
    EEG_start_time_sec=str2num(filename(39:40));
    if EEG_start_time_sec >= 30
       EEG_start_time_min = EEG_start_time_min+60;
    end
    EEG_start_time=EEG_start_time_hour + EEG_start_time_min; %in [sec]
    
    CAM_test_loc = (CAM_time - EEG_start_time);              %in [sec]
    
    % take the part within T hours (both sides) of the CAM test
    T = 3;  % [hour] = 3*60 = 180[min] = 180*60= 10.800 [s]
    Tpoint=T*3600*Fs; %samples
    
    start_loc = max(1, CAM_test_loc*Fs - Tpoint);
    end_loc = min(size(data,2), CAM_test_loc*Fs + Tpoint);
    
    Tmin = 1*60*Fs; % 1min
    if end_loc - start_loc < Tmin
        fprintf('patient %s is ignored due to short length around the CAM test time\n', filename)
        features_final(s,:)=NaN(1,size(features_final,2)); 
        s=s+1;
    else
        
    data = data(1:end, start_loc:end_loc);
%     if isempty(data)
%         continue
%         
%     end
%   lengthdata{s}=[size(data,2) size(data2,2)]; % check if works correctly
%   s=s+1;

    % Zero mean each signal, e.g. remove DC Offset
    med_data= median(data,2);
    zero_data = bsxfun(@minus, data, med_data);
    %clear data med_data;
     
    % Filter data for EEG freqs
    filt_data = FilterEEG(zero_data', Fs, freq_min, freq_max, flag_notch)'; % Data expected in columns rather than rows
    clear zero_data;
    
    % Calculate bipolar leads/refs/montage: standardize extraction
    [bipolar_data, bipolar_names, bipolar_abbrev, unipolar_data, unipolar_names, ref_data] = EEG_CAMS_Leads(filt_data, channels);
    clear filt_data;
    selectdata=bipolar_data([2,4,24,59],:);
    datanames=bipolar_abbrev([2,4,24,59],:);
    clear bipolar_data;
     
    % Resample data when necessary
    resampleddata = EEG_Resample(selectdata, datanames, Fs);
    
    % Segment data into 6s windows   
    % size(data) = (#channel, #points)
    % size(segment) = (#windows, #channel, #points in each window)
    Fs_resamp=200;
    window_size=6*Fs_resamp;
    segs = EEG_CAMS_segmentdata(resampleddata, window_size); 
    %clear resampleddata
    
    % remove segments with artefacts
    artefact = EEG_decide_artifact(segs);
    segs_wa = segs(artefact==0,:,:); % only segments without artefacts selected

%% Calculate spectrograms
%     data_wa=zeros(1,4,size(segs_wa,1)*size(segs_wa,3));           %concatenate data for spectrogram plots          
%     for k = 1:size(segs_wa,1)-1
%         if k == 1
%             data_wa=cat(3,segs_wa(k,:,:),segs_wa(k+1,:,:));
%         else
%         data_wa=cat(3,b,segs_wa(k+1,:,:));
%         end
%         b=data_wa;
%     end 
%      
%     %EEG_CAMS_Spectrogram(data_wa,datanames,Fs,ts,filename); % old spectrogram plot function
%     params.movingwin = [4 1];      % [windowLength stepSize] %
%     params.tapers    = [2 3];      % [TW product No.tapers] %
%     params.fpass     = [0.5 30];   % passband %
%     params.Fs        = 200;        % sampling rate 200Hz %
%     % w_spec = 60*5;               % total duration 10min > take ~7sec %
%     col=[-10 30];                  % range of color scale
%     
%    % [Sdata, stimes, sfreqs] = fcn_computeSpec_avg(squeeze(data_wa), params);    %after artefact removal
%     [Sdata, stimes, sfreqs] = fcn_computeSpec_avg(resampleddata, params);       %raw data
%         colormap jet
%        % d=figure,
%         d=imagesc(stimes, sfreqs, pow2db(Sdata+eps), col);
%         axis xy; box on
%             % to = (round(t_center/Fs)-1)-w_spec/2+1;
%             % t1 = (round(t_center/Fs)-1)+w_spec/2;
%             % tt_spec = timeStampo + seconds(to:60:t1);
%         set(gca, 'xtick', 1:60*2:length(stimes), 'xticklabel', num2cell(0:2:round(length(stimes)/60)));
%         title(filename,'Interpreter','none')
%         ylabel(gca, 'Frequency (Hz)')
%         xlabel(gca, 'Time (min)')
%        % hold on
%        % plot(27*60,25,'vr','MarkerFaceColor','r','MarkerSize',5')
%        colorbar
%        
%        saveas(d,sprintf('Spectrogram_rawdata_%s.png',filename));
    
%% Extract EEG based features
    % Easiest to start with struct and convert to table after?
    % size(temp_features) = (#windows, #features)
    
    % Get time-domain features for each window
      temp_features = [];
      for segs = 1:size(segs_wa,1)
          data2=squeeze(segs_wa(segs,:,:)); 
          temp_features = EEG_CAMS_FeaturesFromEEG_Meike(data2,datanames, 200, filename, tbl_clin);
          features(segs)=temp_features;
      end
      
    % Get Features from PSDs
      f = features(1).f;
      features = EEG_CAMS_FeaturesFromPSD_Meike(features, f);
      
    % Get Features from coherence (also correlations?)
      features=rmfield(features(),{'cohere_Fp1_Fp2_Fp1_Fp2','cohere_Fp1_F7_Fp1_F7','cohere_Fp2_F8_Fp2_F8','cohere_F7_F8_F7_F8'});
      features = EEG_CAMS_FeaturesFromCohere_Meike(features, f);
      features=rmfield(features(),{'cohere_Fp1_Fp2_Fp1_F7','cohere_Fp1_Fp2_Fp2_F8','cohere_Fp1_Fp2_F7_F8','cohere_Fp1_F7_Fp2_F8','cohere_Fp1_F7_F7_F8','cohere_Fp2_F8_F7_F8','psd_db_Fp1_Fp2','psd_db_Fp1_F7','psd_db_Fp2_F8','psd_db_F7_F8','psd_rel_Fp1_Fp2','psd_rel_Fp1_F7','psd_rel_Fp2_F8','psd_rel_F7_F8','psd_val_Fp1_Fp2','psd_val_Fp1_F7','psd_val_Fp2_F8','psd_val_F7_F8','f','ratio_psd_db_thetaVtheta','ratio_psd_rel_thetaVtheta','ratio_sum_cohere_thetaVtheta','ratio_mean_cohere_thetaVtheta'});
     
    % Save extracted features:
      file_save = sprintf('%s%cEEG_CAMS_Features_allsegments-%s', dir_features, filesep, num2str(i_file));
      tic;
      save(file_save);
      fprintf('Done Saving\n');
      toc
        
      % Log transform some features?
      %when is this necessary?
      
      % Make EEG table for classification/regression
      tbl_eeg = struct2table(features);
      
    % take the average of the features from all windows
      tbl_eeg=table2array(tbl_eeg);
      feature_avg = nanmean(tbl_eeg,1);       %  size(temp_features_avg) = (#features,)
        
     % for i = 1:size(feature_avg2,2)
     % eval(['feature_avg2.feature_avg',num2str(i),'=names{i}']);
     % endnames =fieldnames(features);
  
% finally, combine the average features from each patient to form a matrix
% size(features_final) = (#patients, #features) 
     features_final(s,:)=feature_avg; 
     s=s+1;

   TimeUpdate(i_file, numel(files));
   
   clear features
    end
   
end
 
% Save extracted features: features_final (#patient #features)
%       file_save = sprintf('%s%cEEG_CAMS_Features_final-%s', dir_features, filesep, DateTimestamp());
%       tic;
%       save(file_save);
%       fprintf('Done Saving\n');
%       toc      

%% Log transform some features
% sign(x)log(|x|+1): Allow zero and negative values using this transform: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6251659/
field_names =fieldnames(features)';   %names of features
[num_row, num_col] = size(features_final);
tic;
%features_final=array2table(features_final);
%features_final=table2struct(features_final);

features_final_log=LogTransform(features_final);

% for i_f = 1:num_col
%     temp_data = [features_final{:, i_f}];
%     if ~strncmp(field_names{i_f}, 'log_', 4) && size(temp_data, 1) == num_row && 1 == size(temp_data, 2)
%         temp_data = LogTransform(temp_data);
%         temp_name = sprintf('log_%s', field_names{i_f});
%         features_final{:, temp_name} = temp_data;
%     end
%    % TimeUpdate(i_f, num_col, 1e2);
% end

%% Combine clinical and EEG data (CAMS score + features) 
% Pull up clinical data (CAMS score)
datatable_clin=tbl_clin(:,[4, 102, 103]); % subject ID + corrected short CAM-S severity + long CAM-S severity

%tbl_eeg=array2table(features_final);
tbl_eeg=array2table(features_final_log); %for log transformed features
filenames = {files.name};
[a,~, ~, ~] = regexp(filenames, 'DELIRIUM_NON_ICU_AMSD([0-9]{3})', 'tokens');
temp = [a{:}];
subj = str2double(vertcat(temp{:}));
tbl_eeg.SubjectID = subj;           %gives EEG table for included patients

%f = features(1).f;

% Add clinical scores to EEG data
% field_out = {'cam3D_score', 'cam3D_severity', 'cam3D_2', 'CAMSscorealgorithm' ,'camSeverityScore'};
field_out = {'CorrectedShortFormSeverityScore07','CorrectedLongFormSeverityScore019'};
for i_field = 1:numel(field_out)
    temp_field = field_out{i_field};
    tbl_eeg.(temp_field) = nan(size(tbl_eeg, 1), 1);
    temp_out = double(datatable_clin.(temp_field)); % Convert to double in order to have NaN allowable as well
    for i_subj = 1:size(tbl_eeg, 1)
        mask_clin_subj = datatable_clin.SUBJECTID == tbl_eeg{i_subj, 'SubjectID'};
        if sum(mask_clin_subj)
           tbl_eeg{i_subj, temp_field} = temp_out(mask_clin_subj);
        end
    end
end

tbl_eeg=table2array(tbl_eeg);
tbl_eeg=tbl_eeg(all(~isnan(tbl_eeg),2),:); %removes rows with nan values (for eeg's outside time limit (3h) or nan value for CAMS)
tbl_eeg=array2table(tbl_eeg);

CAMSshort=tbl_eeg(:,end-1);         %Outcome measure CAMS (short form)
CAMSlong=tbl_eeg(:,end);            %Outcome measure CAMS (long form)
tbl_eeg_old=tbl_eeg;                %With outcome measures included
tbl_eeg=tbl_eeg(:,1:end-3);         %final eeg feature matrix
%tbl_eeg.Outcome = tbl_eeg.(field_out{1});

%% Correlations (make scatter plots and look at spearman's correlation)
tbl_eeg=table2array(tbl_eeg);
tbl_eeg_old=table2array(tbl_eeg_old);
CAMSshort=table2array(CAMSshort);
CAMSlong=table2array(CAMSlong);
names =fieldnames(features);   %names of features
for u=1:size(tbl_eeg,2);                       
    k=scatter(tbl_eeg(:,u),CAMSshort);
    [rho,pval]=corr(tbl_eeg(:,u),CAMSshort,'Type','Spearman');
    rho(u,:)=rho;
    pval(u,:)=pval;
    ylabel('CAMS score (short)')
    xlabel(sprintf('%s',names{u}),'Interpreter','none');
    saveas(k,sprintf('Scatterplot_CAMSshort_first200_logtransform%s.png',names{u}));
end

for u=1:size(tbl_eeg,2);                       
    %k=scatter(tbl_eeg(:,u),CAMSlong);
    [rho,pval]=corr(tbl_eeg(:,u),CAMSlong,'Type','Spearman');
    rho(u,:)=rho;
    pval(u,:)=pval;
   % ylabel('CAMS score (long)')
   % xlabel(sprintf('%s',names{u}),'Interpreter','none');
   % saveas(k,sprintf('Scatterplot_CAMSlong_first200_logtransform%s.png',names{u}));
end

% col_names = {'RASS'};
% temp_data = tbl_eeg_sub{:, col_names};
[rho,pval] = corr(tbl_eeg_old, 'Type', 'Spearman');
% corrplot(tbl_eeg,'type','Spearman');

%% Boxplots?

%% Search for correlations
out_reg = 'cam3D_severity';
% out_reg = 'cam3D_2';
% out_reg = 'RASS';
% out_reg = 'GCS';
% out_reg = 'DRSScore';
% out_reg = 'Age';
% out_reg = 'LengthofHospitalStay';
% out_reg = 'GOSathospitaldischarge';

% str_target = 'freq_psd_';
% field_names = fieldnames(features);
% idx_features = find(strncmp(field_names, str_target, numel(str_target)));

data_reg = tbl_eeg_sub.(out_reg);
all_r_sub = nan(numel(idx_features), 1);
feature_group = cell(numel(idx_features), 1);
tic;
for i_feature = 1:numel(idx_features)
    temp_data = [tbl_eeg_sub{:, field_names{idx_features(i_feature)}}];
    r = correl(data_reg, temp_data);
    all_r_sub(i_feature) = r;
    temp_str = field_names{idx_features(i_feature)};
    if strncmp(temp_str, 'freq_', 5)
        all_token = regexp(temp_str, 'freq_psd_([a-z]*_[a-zA-Z0-9]*_[a-zA-Z0-9]*)_', 'tokens');
    elseif strncmp(temp_str, 'fooof_', 6)
        all_token = regexp(temp_str, 'fooof_psd_val_([a-z]*_[a-zA-Z0-9]*_[a-zA-Z0-9]*)_', 'tokens');
    end
    if ~isempty(all_token) && ~isempty(all_token{1})
        feature_group{i_feature} = all_token{1}{1};
    end
    TimeUpdate(i_feature, numel(idx_features), 1e2);
end
abs_r_sub = abs(all_r_sub);


%% Joint plot of perf & correl for each and all features
% all_token = regexp(field_names, '(^[a-z0-9]*_[a-z0-9]*_[a-z0-9]*_[A-Za-z0-9]*)_', 'tokens');
all_token = regexp(field_names, '(^[a-z0-9]*_[a-z0-9]*_[a-z0-9]*)_', 'tokens');
str_group = cell(numel(field_names), 1);
for i_f = 1:numel(field_names)
    if ~isempty(all_token{i_f})
        str_group{i_f} = all_token{i_f}{1}{1};
    else
        temp_token = regexp(field_names{i_f}, '(^[a-z0-9]*_[a-z0-9]*)_', 'tokens');
        if ~isempty(temp_token)
            str_group{i_f} = temp_token{1}{1};
        else
            str_group{i_f} = 'na';
        end
    end
end

[sort_group, idx_sort] = sort(str_group);
clear y_tick*;
y_tick_start = 1;
y_tick_label{1} = sort_group{1};
for i_f = 2:numel(sort_group)
    if ~strcmp(sort_group{i_f}, sort_group{i_f-1})
        y_tick_start(end+1) = i_f;
        y_tick_label{end + 1} = sort_group{i_f};
    end
end
y_tick_end = [y_tick_start(2:end)-1, numel(sort_group)];
y_tick_mid = mean([y_tick_start; y_tick_end]);

font_size = 8;
clf;
boundaries = [0.1 0.98 0.08 0.96];
margins = [0.2 0.2];
grid_axes = AxesGrid(1, 2, boundaries, margins);
alpha = 0.1;

axes(grid_axes(1));
hold on;
% Change color based on leads
x_val = abs_auc(idx_sort);
y_val = 1:numel(hash_code);
% h = plot(x_val, y_val, '.');
% h = plot(abs_auc, idx_sort, '.');
for i_code = 1:max(hash_code)
    mask_code = hash_code == i_code;
    idx_a = find(regions == name_code(i_code, 1));
    idx_b = find(regions == name_code(i_code, 2));
    if ~isempty(idx_a) && ~isempty(idx_b)
        temp_color = all_color(idx_a, idx_b, :);
    else
        temp_color = ColorPicker('lightgray');
    end
    h = plot(x_val(mask_code(idx_sort)), y_val(mask_code(idx_sort)), '.');
    set(h, 'Color', [temp_color(:)', alpha]);
end

title(sprintf('%s: Perf Curve AUC (n=%d features)', out_class, numel(abs_auc)));
x_lim = [0.5 1];
axis([x_lim, 0.5 numel(abs_auc)+0.5]);
set(gca, 'YDir', 'reverse');
set(gca, 'YTick', y_tick_mid, 'YTickLabel', y_tick_label);
set(gca, 'TickLabelInterpreter', 'none');
set(gca, 'FontSize', font_size);
h_line = line(x_lim', repmat(y_tick_end(1:end-1)-0.5, 2, 1));
set(h_line, 'color', ColorPicker('lightgray'));

axes(grid_axes(2));
hold on;
% Change color based on leads
x_val = abs_r(idx_sort);
y_val = 1:numel(hash_code);
% h = plot(x_val, y_val, '.');
% h = plot(abs_auc, idx_sort, '.');
for i_code = 1:max(hash_code)
    mask_code = hash_code == i_code;
    idx_a = find(regions == name_code(i_code, 1));
    idx_b = find(regions == name_code(i_code, 2));
    if ~isempty(idx_a) && ~isempty(idx_b)
        temp_color = all_color(idx_a, idx_b, :);
    else
        temp_color = ColorPicker('lightgray');
    end
    h = plot(x_val(mask_code(idx_sort)), y_val(mask_code(idx_sort)), '.');
    set(h, 'Color', [temp_color(:)', alpha]);
end

title(sprintf('%s: Correlation (n=%d features)', out_reg, numel(abs_r)));
x_lim = [0 1];
axis([x_lim, 0.5 numel(abs_auc)+0.5]);
set(gca, 'YDir', 'reverse');
set(gca, 'YTick', y_tick_mid, 'YTickLabel', y_tick_label);
set(gca, 'TickLabelInterpreter', 'none');
set(gca, 'FontSize', font_size);
h_line = line(x_lim', repmat(y_tick_end(1:end-1)-0.5, 2, 1));
set(h_line, 'color', ColorPicker('lightgray'));

% Print some top features
field_names = tbl_eeg_sub.Properties.VariableNames;
[val_auc, idx_auc] = sort(abs_auc, 'descend','MissingPlacement','last');
[val_r, idx_r] = sort(abs_r, 'descend','MissingPlacement','last');
num_val = 50;
for i_val = 1:num_val
    fprintf('%3d ', i_val);
    fprintf('\t');
    fprintf('%-50s', field_names{idx_auc(i_val)});
    fprintf('\t');
    fprintf('%0.3f', all_auc(idx_auc(i_val)));
    fprintf('\t');
    fprintf('%-50s', field_names{idx_r(i_val)});
    fprintf('\t');
    fprintf('% 0.3f', all_r(idx_r(i_val)));
    fprintf('\n');
end
fprintf('\n');


%% Scatter plot of both clinical and EEG del sev by outcome/Death
clf;
hold on;
colors = ColorGradientBlueDarkGrayRed(8);
colors = ColormapGradient2(ColorPicker('blue'), ColorPicker('red'), 1, 8);
colors = [ColorPicker('gray'); ColorPicker('red')];
[name_group, ~, hash_group] = unique(tbl_eeg_sub.Death);
markers = 'ox';
for i_group = 1:max(hash_group)
    mask_group = hash_group == i_group;
    for i_c = min(tbl_eeg_sub.cam3D_severity):max(tbl_eeg_sub.cam3D_severity)
        mask_sev = tbl_eeg_sub.cam3D_severity == i_c;
        if sum(mask_group & mask_sev)
            %             h = plot(i_c, tbl_eeg_sub{mask_group & mask_sev, 'RegDelEEG'}, '.');
            %             set(h, 'Color', [colors(i_group, :) 0.1]);
            %             set(h, 'MarkerSize', 20);
            mask = mask_group & mask_sev;
            h = scatter(i_c*ones(sum(mask), 1)+randn(sum(mask), 1)/20, tbl_eeg_sub{mask, 'RegDelEEG'}, 'filled');
            %             h = scatter(i_c*ones(sum(mask), 1)+randn(sum(mask), 1)/20, tbl_eeg_sub{mask, 'RegDelEEG'});
            %             set(h, 'Marker', markers(i_group));
            %             set(h, 'MarkerEdgeColor', colors(i_group, :));
            %             set(h, 'MarkerEdgeAlpha', 0.5);
            set(h, 'MarkerFaceColor', colors(i_group, :));
            set(h, 'MarkerFaceAlpha', 0.5);
            set(h, 'SizeData', 60);
            
            %             set(h, 'Marker', 'o');
        end
        %         if sum(mask_group & mask_sev)
        %             h = Violin(tbl_eeg_sub{mask_group & mask_sev, 'RegDelEEG'}, i_c);
        %             h.ViolinColor = colors(i_group, :);
        %         end
    end
end
% axis([-inf inf 0 inf]);
axis([-0.5 max_score + 0.5 0 inf]);
xlabel('Clinical Delirium Severity');
ylabel('EEG-predicted Delirium Severity');
% title('EEG-Predicted Delirium Severity vs. Clinically-assessed Delirium Severity');
set(gca, 'Fontsize', 16);

axis square
r = corrcoef(double(ytes), yptes);
title(sprintf('R = %.3f, R2 = %.3f', r(2), r(2)^2));

%% Plot EEG data
% plot(ts/60,data)
% xlabel('time(min)')
% ylabel('mV')
% title('Raw data')

