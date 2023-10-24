clear all; clc; format compact;

%Modifying Meike's LTR model for ICANS
%set(0,'DefaultFigureWindowStyle','docked') %figures in tab

%% Path settings for script
% Dependencies copied into shared dropbox folder 2019/12/23: add current path and set paths in relation to it
mfile_full = matlab.desktop.editor.getActiveFilename;
[dir_mfile, mfile] = fileparts(mfile_full);
addpath(dir_mfile);

cell_dir = split(dir_mfile, filesep);
% %dir_clin = fullfile(cell_dir{1:end-1}, filesep, 'DataClinical'); %Clinical data should be csv file with ICANS scores
% dir_spec = fullfile(cell_dir{1:end-1}, filesep, 'Spectrograms');
% %dir_code = fullfile(cell_dir{1:end-1}, filesep, 'CodeEyal');
% dir_eeg = fullfile(cell_dir{1:end-1}, filesep, 'DataEEG');
dir_features = fullfile(cell_dir{1:end-1}, filesep, 'DataFeatures2');



%% Load EEG data for individuals, already processed (filter, artifact labels) in Python
dirname = 'D:\EEGs\EEGSegments1';
files=dir('*.mat')
% https://www-sciencedirect-com.ezp-prod1.hul.harvard.edu/science/article/pii/S0165027015000230

%s=1; %counter

%%
        cd(dirname)%cd(dir_eeg2);
        files=dir('*.mat')
for i_file = 1%:numel(files) %ook nog vanaf 2:5?
    
%     if subj(i_file) == subj(i_file-1)
%         continue
%     else
        
        % Load data (and concatenate data when necessary, when multiple recordings)
        cd(dirname)%cd(dir_eeg2);
        files=dir('*.mat')
        filename = files(i_file).name;
        load(filename,'Data','Fs','Start','ICANS','ICE','SourceFile','sid')       
        bipolar_abbrev = {'Fp1_F7', 'Fp1_F3', 'Fp2_F4', 'Fp2_F8'};
        subjectnr=sid
        source=strsplit(SourceFile,".")
        source=source{1}

 %   s=number;

    % Zero mean each signal, e.g. remove DC Offset
 %%   med_data= median(data,2);
 %   zero_data = bsxfun(@minus, data, med_data);
  %  %clear data med_data;
     

% % Calculate spectrograms
% %     data_wa=zeros(1,4,size(segs_wa,1)*size(segs_wa,3));           %concatenate data for spectrogram plots          
% %     for k = 1:size(segs_wa,1)-1
% %         if k == 1
% %             data_wa=cat(3,segs_wa(k,:,:),segs_wa(k+1,:,:));
% %         else
% %         data_wa=cat(3,b,segs_wa(k+1,:,:));
% %         end
% %         b=data_wa;
% %     end 
% %      
% %     %EEG_CAMS_Spectrogram(data_wa,datanames,Fs,ts,filename); % old spectrogram plot function
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
%         d=imagesc(stimes, sfreqs, pow2db(Sdata+eps), col);
%         axis xy; box on
%             % to = (round(t_center/Fs)-1)-w_spec/2+1;
%             % t1 = (round(t_center/Fs)-1)+w_spec/2;
%             % tt_spec = timeStampo + seconds(to:60:t1);
%         set(gca, 'xtick', 1:60*60:length(stimes), 'xticklabel', num2cell(0:1:round(length(stimes)/60/60)));
%         title(['Subject ',num2str(subj(i_file)),''],'Interpreter','none')
%         ylabel(gca, 'Frequency (Hz)')
%         xlabel(gca, 'Time (hours)')
%        hold on
%        plot(CAM_test_loc,25,'vr','MarkerFaceColor','r','MarkerSize',5') %Clinical test time location
%        colorbar
%        
%        dir_eeg2='Z:\Projects\Meike';
%        cd(dir_eeg2)
%        saveas(d,sprintf('Spectrogram_rawdata_%s.png',filename));
%     end
    
% Extract EEG based features
    % Easiest to start with struct and convert to table after?
    % size(temp_features) = (#windows, #features)
    
    % Get time-domain features for each window
      temp_features = [];
      parfor segs = 1:size(Data,1)
          data2=squeeze(Data(segs,:,:)); 
          temp_features = EEG_ICANS_FeaturesFromEEG_Meike(data2,bipolar_abbrev, 200, filename);
          features(segs)=temp_features;
      end
      
    % Get Features from PSDs
      f = features(1).f;
      features = EEG_ICANS_FeaturesFromPSD_Meike(features, f);
      
    % Get Features from coherence (also correlations?)
      features=rmfield(features(),{'cohere_Fp1_F3_Fp1_F3','cohere_Fp1_F7_Fp1_F7','cohere_Fp2_F8_Fp2_F8','cohere_Fp2_F4_Fp2_F4'});
      features = EEG_CAMS_FeaturesFromCohere_Meike(features, f);
     % features=rmfield(features(),{'cohere_Fp1_F7_Fp1_F3'})
      features=rmfield(features(),{'cohere_Fp1_F3_Fp2_F4','cohere_Fp1_F3_Fp2_F8','cohere_Fp2_F4_Fp2_F8','cohere_Fp1_F7_Fp1_F3','cohere_Fp1_F7_Fp2_F4','cohere_Fp1_F7_Fp2_F8','psd_val_Fp1_F7','psd_rel_Fp2_F8','psd_db_Fp1_F7','psd_rel_Fp1_F7','psd_val_Fp1_F3','psd_db_Fp1_F3','psd_rel_Fp1_F3','psd_val_Fp2_F4','psd_db_Fp2_F4','psd_rel_Fp2_F4','psd_val_Fp2_F8','psd_db_Fp2_F8','f','ratio_psd_db_thetaVtheta','ratio_psd_rel_thetaVtheta','ratio_sum_cohere_thetaVtheta','ratio_mean_cohere_thetaVtheta'});
     
    % Save extracted features:
    ending = strcat("_",num2str(subjectnr),'_',int2str(ICANS),'_',int2str(ICE),"_",source)
      file_save = sprintf('%s%cEEG_ICANS_Features_allsegments-subject%s', dir_features, filesep, ending);
      save(file_save,'features');
      
    % Log transform some features
      features2=struct2table(features); features2=table2array(features2);
      features_log_all=LogTransform(features2);
      features_log_selection1=LogTransform(features2(:,[1:71]));
      features_log_selection2=LogTransform(features2(:,[125:end]));
      features_log_selection=[features_log_selection1, features2(:,72:124),features_log_selection2];
      file_save = sprintf('%s%cEEG_ICANS_Features_log_allsegments-subject%s', dir_features, filesep, ending);
      save(file_save,'features_log_all');
      file_save = sprintf('%s%cEEG_ICANS_Features_log_select_allsegments-subject%s', dir_features, filesep, ending);
      save(file_save,'features_log_selection');
      names=fieldnames(features);    

   TimeUpdate(i_file, numel(files));
   
   clear source sourceFile Start ICANS ICE features features2 features_log_all features_log_selection features_log_selection1 features_log_selection2 feature_tbl feature_tbl_log feature_tbl_log_select
   
    %% Reload data per patient
       cd(dir_features)
       files=dir('*.mat')
subjects=size(files,1);     %it's actually the number of files, not number of subjects  
num_features=293;
features_avg=NaN(size(subjects,1),num_features); features_avg_log=NaN(size(subjects,1),num_features); features_avg_log2=NaN(size(subjects,1),num_features);
features_std=NaN(size(subjects,1),num_features); features_std_log=NaN(size(subjects,1),num_features); features_std_log2=NaN(size(subjects,1),num_features);
features_min=NaN(size(subjects,1),num_features); features_min_log=NaN(size(subjects,1),num_features); features_min_log2=NaN(size(subjects,1),num_features);
features_max=NaN(size(subjects,1),num_features); features_max_log=NaN(size(subjects,1),num_features); features_max_log2=NaN(size(subjects,1),num_features);
%%
      % Find files
       featurefiles = dir('EEG_ICANS_Features_allsegments-subject*.mat');
       featurefileslog=dir('EEG_ICANS_Features_log_allsegments-subject*.mat');
       featurefileslog2=dir('EEG_ICANS_Features_log_select_allsegments-subject*.mat');
       featurefilenames={featurefiles.name};
       featurefilenameslog={featurefileslog.name};
       featurefilenameslog2={featurefileslog2.name};
       
     for i_filef = 1:numel(featurefilenameslog)
            
            featurefilename = featurefiles(i_filef).name
            featurefilenamelog = featurefileslog(i_filef).name;
            featurefilenamelog2 = featurefileslog2(i_filef).name;
          %  Load data
            load(featurefilename);      %features
            load(featurefilenamelog);    %log features
            load(featurefilenamelog2);  %log2 features
 
      % Make EEG table for classification/regression
      feature_tbl = struct2table(features); feature_tbl=table2array(feature_tbl);
      feature_tbl_log=features_log_all;
      feature_tbl_log_select=features_log_selection;
      
    % take the average of the features  from all windows
      % take the average, sd, min and max of the features from all windows   % size(temp_features_avg) = (#features,)
      feature_avg = nanmean(feature_tbl,1);       
      feature_std=std(feature_tbl,1);
      feature_min=min(feature_tbl,[],1);
      feature_max=max(feature_tbl,[],1);
      feature_avg_log=nanmean(feature_tbl_log,1);       
      feature_std_log=std(feature_tbl_log,1);
      feature_min_log=min(feature_tbl_log,[],1);
      feature_max_log=max(feature_tbl_log,[],1);
      feature_avg_log2=nanmean(feature_tbl_log_select,1);       
      feature_std_log2=std(feature_tbl_log_select,1);
      feature_min_log2=min(feature_tbl_log_select,[],1);
      feature_max_log2=max(feature_tbl_log_select,[],1);
      
     % [a2] = regexp(featurefilename, 'EEG_CAMS_ICANS_Features_allsegments-subject([0-9]{3})', 'tokens');
    %  temp = [a2{:}];
    %  subjectnr = sid; %subject ID
    %  number = find(subjects==subjectnr);       %row number
    s=i_filef;
  
% finally, combine the average features from each patient to form a matrix
% size(features_final) = (#patients, #features) 
     features_avg(s,:)=feature_avg; features_avg_log(s,:)=feature_avg_log; features_avg_log2(s,:)=feature_avg_log2;
     features_std(s,:)=feature_std; features_std_log(s,:)=feature_std_log; features_std_log2(s,:)=feature_std_log2;
     features_min(s,:)=feature_min; features_min_log(s,:)=feature_min_log; features_min_log2(s,:)=feature_min_log2;
     features_max(s,:)=feature_max; features_max_log(s,:)=feature_max_log; features_max_log2(s,:)=feature_max_log2;

         
         
     contents=strsplit(featurefilename,"_");
     sid=contents{5};
     ICANS=contents{6};
     ICE=contents{7};
     source_file=strcat(contents{8:end});
 

    if size(sid,2)==1
%          sid=contents{4}(20)
           sid=strcat("00",sid);
    elseif size(sid, 2)== 2
        sid=strcat("0",sid);
    end
    
    if size(ICE,2)==1
        ICE=strcat("0",ICE);
    end
    if size(source_file,2) ~= 69
        diff = 69-size(source_file,2);
        source_file=strcat(repmat('0',1,diff),source_file);
    end
    size(source_file,2);

      SID_tot(s,:)=sid;
      file_tot(s,:)=source_file;
      ICE_tot(s,:)=ICE;
      ICANS_tot(s,:)=ICANS;
     
   %TimeUpdate(i_filef, numel(featurefiles));

   clear features contents sid ICE ICANS source_file  
       end
   end
%  end
%end

%% Final feature matrix
 names=names';
 names_avg = [];
 names_std = [];
 names_min = [];
 names_max = [];
for i = 1:length(names)
  names_avg{i} = ['avg_',names{i}];
  names_std{i} = ['std_',names{i}];
  names_min{i} = ['min_',names{i}];
  names_max{i} = ['max_',names{i}];
end
feature_names=[names_avg, names_std, names_min, names_max];

features_final_ICANS=[features_avg,features_std,features_min,features_max];          %289*4 = 1156 298*4=1192 features
features_final_ICANS=array2table(features_final_ICANS,'VariableNames',feature_names);      %feature table
%features_final_ICANS=table2array(features_final_ICANS); 

features_final_ICANS_log=[features_avg_log,features_std_log,features_min_log,features_max_log];
features_final_ICANS_log=array2table(features_final_ICANS_log,'VariableNames',feature_names);
%features_final_ICANS_log=table2array(features_final_ICANS_log); 

features_final_ICANS_log2=[features_avg_log2,features_std_log2,features_min_log2,features_max_log2];
features_final_ICANS_log2=array2table(features_final_ICANS_log2,'VariableNames',feature_names);
%features_final_ICANS_log2=table2array(features_final_ICANS_log2);

file_save = sprintf('features_final_ICANS', dir_features, filesep);
      save(file_save,'features_final_ICANS');
file_save = sprintf('features_final_ICANS_log', dir_features, filesep);
      save(file_save,'features_final_ICANS_log');
file_save = sprintf('features_final_ICANS_log2', dir_features, filesep);
      save(file_save,'features_final_ICANS_log2');

features_final_array=table2array(features_final_ICANS);
features_final_afterlog_ICANS=LogTransform(features_final_array);
features_final_afterlog_ICANS=array2table(features_final_afterlog_ICANS,'VariableNames',feature_names);

file_save = sprintf('features_final_afterlog_ICANS', dir_features, filesep);
      save(file_save,'features_final_afterlog_ICANS');
%      
% %% Extract CAMS scores + subject IDs
% % add cams scores + subject IDs to feature matrix
% %feature_names2=feature_names(1,[1:481,485:end]);
% 
% % select only the rows without nans (so with cams scores)
% features_final_ICANS=features_final_ICANS(:,[1:481,485:end]);
% features_final_ICANS.SubjectID = subjectIDs;
% features_final_ICANS.CAMSscores = CAMSscores;
% feature_names2 = fieldnames(features_final_ICANS)';   %names of features
% feature_names2 = feature_names2(1,1:end-2);
% features_final_ICANS=table2array(features_final_ICANS);
% features_final_ICANS(any(isnan(features_final_ICANS),2),:) = [];
% features_final_ICANS=array2table(features_final_ICANS,'VariableNames',feature_names2);
% features1=features_final_ICANS(:,1:end-2);
% 
% features_final_ICANS_log=features_final_ICANS_log(:,[1:481,485:end]);
% features_final_ICANS_log.SubjectID = subjectIDs;
% features_final_ICANS_log.CAMSscores = CAMSscores;
% features_final_ICANS_log=table2array(features_final_ICANS_log);
% features_final_ICANS_log(any(isnan(features_final_ICANS_log),2),:) = [];
% features_final_ICANS_log=array2table(features_final_ICANS_log,'VariableNames',feature_names2);
% features2=features_final_ICANS_log(:,1:end-2);
% 
% features_final_afterlog_ICANS=features_final_afterlog_ICANS(:,[1:481,485:end]);
% features_final_afterlog_ICANS.SubjectID = subjectIDs;
% features_final_afterlog_ICANS.CAMSscores = CAMSscores;
% features_final_afterlog_ICANS=table2array(features_final_afterlog_ICANS);
% features_final_afterlog_ICANS(any(isnan(features_final_afterlog_ICANS),2),:) = [];
% features_final_afterlog_ICANS=array2table(features_final_afterlog_ICANS,'VariableNames',feature_names2);
% features4=features_final_afterlog_ICANS(:,1:end-2);
% 
% %%
% features_final_ICANS_log2=features_final_ICANS_log2(:,[1:481,485:end]);
% features_final_ICANS_log2.SubjectID = subjectIDs;
% features_final_ICANS_log2.CAMSscores = CAMSscores;
% features_final_ICANS_log2.Coma = comascores; 
% %feature_names2 = fieldnames(features_final_ICANS2)';   %names of features
% %feature_names2 = feature_names2(1,1:end-3);
% features_final_ICANS_log2=table2array(features_final_ICANS_log2);
% features_final_ICANS_log2(any(isnan(features_final_ICANS_log2),2),:) = [];
% ids = features_final_ICANS_log2(:,1192) == 0;
% features_ICANS = features_final_ICANS_log2(ids == 1,:);
% 
% features_ICANS_excl=features_nonICANS(:,1:end-3); %change to features_ICANS!
% features_ICANS_excl=array2table(features_ICANS_excl,'VariableNames',feature_names_final);
% 
% % Only for log2 data after excluding comatose patients:
% CAMS_ICANS_excl = features_nonICANS(:,end-1);
% subjectIDs_ICANS_excl=features_nonICANS(:,end-2);
% feature_names_final=feature_names2(1,1:end-2); %(based on old names)
% %% Log transform some features
% % % sign(x)log(|x|+1): Allow zero and negative values using this transform: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6251659/
% % field_names =fieldnames(features)';   %names of features
% % [num_row, num_col] = size(features_final);
% % tic;
% % %features_final=array2table(features_final);
% % %features_final=table2struct(features_final);
% % 
% % features_final_log=LogTransform(features_final);
% 
% %% Combine clinical and EEG data (CAMS score + features) --> need to check below to extract CAMS scores
% % Pull up clinical data (CAMS score)
% datatable_clin=tbl_clin(:,[4, 102, 103]); % subject ID + corrected short CAM-S severity + long CAM-S severity
% 
% tbl_eeg=array2table(features_final_log);
% filenames = {files.name};
% [a,~, ~, ~] = regexp(filenames, 'DELIRIUM_ICANS_AMSD([0-9]{3})', 'tokens');
% temp = [a{:}];
% subj = str2double(vertcat(temp{:}));
% tbl_eeg.SubjectID = subj;
% 
% %f = features(1).f;
% % field_out = {'cam3D_score', 'cam3D_severity', 'cam3D_2', 'CAMSscorealgorithm' ,'camSeverityScore'};
% field_out = {'CorrectedShortFormSeverityScore07','CorrectedLongFormSeverityScore019'};
% for i_field = 1:numel(field_out)
%     temp_field = field_out{i_field};
%     tbl_eeg.(temp_field) = nan(size(tbl_eeg, 1), 1);
%     temp_out = double(datatable_clin.(temp_field)); % Convert to double in order to have NaN allowable as well
%     for i_subj = 1:size(tbl_eeg, 1)
%         mask_clin_subj = datatable_clin.SUBJECTID == tbl_eeg{i_subj, 'SubjectID'};
%         if sum(mask_clin_subj)
%            tbl_eeg{i_subj, temp_field} = temp_out(mask_clin_subj);
%         end
%     end
% end
% 
% tbl_eeg=table2array(tbl_eeg);
% tbl_eeg=tbl_eeg(all(~isnan(tbl_eeg),2),:); %removes rows with nan values (for eeg's outside time limit (3h) or nan value for CAMS)
% tbl_eeg=array2table(tbl_eeg);
% 
% CAMSshort=tbl_eeg(:,end-1);
% CAMSlong=tbl_eeg(:,end);
% tbl_eeg=tbl_eeg(:,1:end-3);
% %tbl_eeg.Outcome = tbl_eeg.(field_out{1});
% 
% %% Correlations (make scatter plots and look at spearman's correlation)
% tbl_eeg=table2array(tbl_eeg);
% CAMSshort=table2array(CAMSshort);
% CAMSlong=table2array(CAMSlong);
% names =fieldnames(features);   %names of features
% for u=1:size(tbl_eeg,2);                       
%     k=scatter(tbl_eeg(:,u),CAMSshort);
%     [rho,pval]=corr(tbl_eeg(:,u),CAMSshort,'Type','Spearman');
%     rho(u,:)=rho;
%     pval(u,:)=pval;
%     ylabel('CAMS score (short)')
%     xlabel(sprintf('%s',names{u}),'Interpreter','none');
%     saveas(k,sprintf('Scatterplot_logtransform_CAMSshort_%s.png',names{u}));
% end
% 
% for u=1:size(tbl_eeg,2);                       
%     k=scatter(tbl_eeg(:,u),CAMSlong);
%     [rho,pval]=corr(tbl_eeg(:,u),CAMSlong,'Type','Spearman');
%     rho(u,:)=rho;
%     pval(u,:)=pval;
%     ylabel('CAMS score (long)')
%     xlabel(sprintf('%s',names{u}),'Interpreter','none');
%     saveas(k,sprintf('Scatterplot_logtransform_CAMSlong_%s.png',names{u}));
% end
% 
% % col_names = {'RASS'};
% % temp_data = tbl_eeg_sub{:, col_names};
% % [rho,pval] = corr(temp_data, 'Type', 'Spearman');
% % corrplot(tbl_eeg,'type','Spearman');
% 
% %% Boxplots?
% 
% %% Search for correlations
% out_reg = 'cam3D_severity';
% % out_reg = 'cam3D_2';
% % out_reg = 'RASS';
% % out_reg = 'GCS';
% % out_reg = 'DRSScore';
% % out_reg = 'Age';
% % out_reg = 'LengthofHospitalStay';
% % out_reg = 'GOSathospitaldischarge';
% 
% % str_target = 'freq_psd_';
% % field_names = fieldnames(features);
% % idx_features = find(strncmp(field_names, str_target, numel(str_target)));
% 
% data_reg = tbl_eeg_sub.(out_reg);
% all_r_sub = nan(numel(idx_features), 1);
% feature_group = cell(numel(idx_features), 1);
% tic;
% for i_feature = 1:numel(idx_features)
%     temp_data = [tbl_eeg_sub{:, field_names{idx_features(i_feature)}}];
%     r = correl(data_reg, temp_data);
%     all_r_sub(i_feature) = r;
%     temp_str = field_names{idx_features(i_feature)};
%     if strncmp(temp_str, 'freq_', 5)
%         all_token = regexp(temp_str, 'freq_psd_([a-z]*_[a-zA-Z0-9]*_[a-zA-Z0-9]*)_', 'tokens');
%     elseif strncmp(temp_str, 'fooof_', 6)
%         all_token = regexp(temp_str, 'fooof_psd_val_([a-z]*_[a-zA-Z0-9]*_[a-zA-Z0-9]*)_', 'tokens');
%     end
%     if ~isempty(all_token) && ~isempty(all_token{1})
%         feature_group{i_feature} = all_token{1}{1};
%     end
%     TimeUpdate(i_feature, numel(idx_features), 1e2);
% end
% abs_r_sub = abs(all_r_sub);
% 
% 
% %% Joint plot of perf & correl for each and all features
% % all_token = regexp(field_names, '(^[a-z0-9]*_[a-z0-9]*_[a-z0-9]*_[A-Za-z0-9]*)_', 'tokens');
% all_token = regexp(field_names, '(^[a-z0-9]*_[a-z0-9]*_[a-z0-9]*)_', 'tokens');
% str_group = cell(numel(field_names), 1);
% for i_f = 1:numel(field_names)
%     if ~isempty(all_token{i_f})
%         str_group{i_f} = all_token{i_f}{1}{1};
%     else
%         temp_token = regexp(field_names{i_f}, '(^[a-z0-9]*_[a-z0-9]*)_', 'tokens');
%         if ~isempty(temp_token)
%             str_group{i_f} = temp_token{1}{1};
%         else
%             str_group{i_f} = 'na';
%         end
%     end
% end
% 
% [sort_group, idx_sort] = sort(str_group);
% clear y_tick*;
% y_tick_start = 1;
% y_tick_label{1} = sort_group{1};
% for i_f = 2:numel(sort_group)
%     if ~strcmp(sort_group{i_f}, sort_group{i_f-1})
%         y_tick_start(end+1) = i_f;
%         y_tick_label{end + 1} = sort_group{i_f};
%     end
% end
% y_tick_end = [y_tick_start(2:end)-1, numel(sort_group)];
% y_tick_mid = mean([y_tick_start; y_tick_end]);
% 
% font_size = 8;
% clf;
% boundaries = [0.1 0.98 0.08 0.96];
% margins = [0.2 0.2];
% grid_axes = AxesGrid(1, 2, boundaries, margins);
% alpha = 0.1;
% 
% axes(grid_axes(1));
% hold on;
% % Change color based on leads
% x_val = abs_auc(idx_sort);
% y_val = 1:numel(hash_code);
% % h = plot(x_val, y_val, '.');
% % h = plot(abs_auc, idx_sort, '.');
% for i_code = 1:max(hash_code)
%     mask_code = hash_code == i_code;
%     idx_a = find(regions == name_code(i_code, 1));
%     idx_b = find(regions == name_code(i_code, 2));
%     if ~isempty(idx_a) && ~isempty(idx_b)
%         temp_color = all_color(idx_a, idx_b, :);
%     else
%         temp_color = ColorPicker('lightgray');
%     end
%     h = plot(x_val(mask_code(idx_sort)), y_val(mask_code(idx_sort)), '.');
%     set(h, 'Color', [temp_color(:)', alpha]);
% end
% 
% title(sprintf('%s: Perf Curve AUC (n=%d features)', out_class, numel(abs_auc)));
% x_lim = [0.5 1];
% axis([x_lim, 0.5 numel(abs_auc)+0.5]);
% set(gca, 'YDir', 'reverse');
% set(gca, 'YTick', y_tick_mid, 'YTickLabel', y_tick_label);
% set(gca, 'TickLabelInterpreter', 'none');
% set(gca, 'FontSize', font_size);
% h_line = line(x_lim', repmat(y_tick_end(1:end-1)-0.5, 2, 1));
% set(h_line, 'color', ColorPicker('lightgray'));
% 
% axes(grid_axes(2));
% hold on;
% % Change color based on leads
% x_val = abs_r(idx_sort);
% y_val = 1:numel(hash_code);
% % h = plot(x_val, y_val, '.');
% % h = plot(abs_auc, idx_sort, '.');
% for i_code = 1:max(hash_code)
%     mask_code = hash_code == i_code;
%     idx_a = find(regions == name_code(i_code, 1));
%     idx_b = find(regions == name_code(i_code, 2));
%     if ~isempty(idx_a) && ~isempty(idx_b)
%         temp_color = all_color(idx_a, idx_b, :);
%     else
%         temp_color = ColorPicker('lightgray');
%     end
%     h = plot(x_val(mask_code(idx_sort)), y_val(mask_code(idx_sort)), '.');
%     set(h, 'Color', [temp_color(:)', alpha]);
% end
% 
% title(sprintf('%s: Correlation (n=%d features)', out_reg, numel(abs_r)));
% x_lim = [0 1];
% axis([x_lim, 0.5 numel(abs_auc)+0.5]);
% set(gca, 'YDir', 'reverse');
% set(gca, 'YTick', y_tick_mid, 'YTickLabel', y_tick_label);
% set(gca, 'TickLabelInterpreter', 'none');
% set(gca, 'FontSize', font_size);
% h_line = line(x_lim', repmat(y_tick_end(1:end-1)-0.5, 2, 1));
% set(h_line, 'color', ColorPicker('lightgray'));
% 
% % Print some top features
% field_names = tbl_eeg_sub.Properties.VariableNames;
% [val_auc, idx_auc] = sort(abs_auc, 'descend','MissingPlacement','last');
% [val_r, idx_r] = sort(abs_r, 'descend','MissingPlacement','last');
% num_val = 50;
% for i_val = 1:num_val
%     fprintf('%3d ', i_val);
%     fprintf('\t');
%     fprintf('%-50s', field_names{idx_auc(i_val)});
%     fprintf('\t');
%     fprintf('%0.3f', all_auc(idx_auc(i_val)));
%     fprintf('\t');
%     fprintf('%-50s', field_names{idx_r(i_val)});
%     fprintf('\t');
%     fprintf('% 0.3f', all_r(idx_r(i_val)));
%     fprintf('\n');
% end
% fprintf('\n');
% 
% colorbar
% %% Scatter plot of both clinical and EEG del sev by outcome/Death
% clf;
% hold on;
% colors = ColorGradientBlueDarkGrayRed(8);
% colors = ColormapGradient2(ColorPicker('blue'), ColorPicker('red'), 1, 8);
% colors = [ColorPicker('gray'); ColorPicker('red')];
% [name_group, ~, hash_group] = unique(tbl_eeg_sub.Death);
% markers = 'ox';
% for i_group = 1:max(hash_group)
%     mask_group = hash_group == i_group;
%     for i_c = min(tbl_eeg_sub.cam3D_severity):max(tbl_eeg_sub.cam3D_severity)
%         mask_sev = tbl_eeg_sub.cam3D_severity == i_c;
%         if sum(mask_group & mask_sev)
%             %             h = plot(i_c, tbl_eeg_sub{mask_group & mask_sev, 'RegDelEEG'}, '.');
%             %             set(h, 'Color', [colors(i_group, :) 0.1]);
%             %             set(h, 'MarkerSize', 20);
%             mask = mask_group & mask_sev;
%             h = scatter(i_c*ones(sum(mask), 1)+randn(sum(mask), 1)/20, tbl_eeg_sub{mask, 'RegDelEEG'}, 'filled');
%             %             h = scatter(i_c*ones(sum(mask), 1)+randn(sum(mask), 1)/20, tbl_eeg_sub{mask, 'RegDelEEG'});
%             %             set(h, 'Marker', markers(i_group));
%             %             set(h, 'MarkerEdgeColor', colors(i_group, :));
%             %             set(h, 'MarkerEdgeAlpha', 0.5);
%             set(h, 'MarkerFaceColor', colors(i_group, :));
%             set(h, 'MarkerFaceAlpha', 0.5);
%             set(h, 'SizeData', 60);
%             
%             %             set(h, 'Marker', 'o');
%         end
%         %         if sum(mask_group & mask_sev)
%         %             h = Violin(tbl_eeg_sub{mask_group & mask_sev, 'RegDelEEG'}, i_c);
%         %             h.ViolinColor = colors(i_group, :);
%         %         end
%     end
% end
% % axis([-inf inf 0 inf]);
% axis([-0.5 max_score + 0.5 0 inf]);
% xlabel('Clinical Delirium Severity');
% ylabel('EEG-predicted Delirium Severity');
% % title('EEG-Predicted Delirium Severity vs. Clinically-assessed Delirium Severity');
% set(gca, 'Fontsize', 16);
% 
% axis square
% r = corrcoef(double(ytes), yptes);
% title(sprintf('R = %.3f, R2 = %.3f', r(2), r(2)^2));
% 
% %% Plot EEG data
% % plot(ts/60,data)
% % xlabel('time(min)')
% % ylabel('mV')
% % title('Raw data')

