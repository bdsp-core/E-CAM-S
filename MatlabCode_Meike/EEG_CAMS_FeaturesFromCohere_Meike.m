%% Calculation of features that depend on full time
function features = EEG_CAMS_FeaturesFromCohere_Meike(features, f)

%% Pull out frequency band info
[freq_bands, ratio_bands] = EEGFreqBandsVaried();
feature_names = fieldnames(features);
freq_bands_names = fieldnames(freq_bands);
ratio_bands_names = fieldnames(ratio_bands);
targets = {'cohere'}; % coherence
temp_data1=zeros(size(features,2),4,26);
temp_data2=zeros(size(features,2),4,26);
temp_data3=zeros(size(features,2),4,28);
temp_data4=zeros(size(features,2),4,28);
for i_target = 1:numel(targets)
    str_target = targets{i_target};
    sub_feature = feature_names(strncmp(feature_names, str_target, numel(str_target)));
    for i_feature = 1:numel(sub_feature)
        % Calculate results within bands
        for i_freq = 1:numel(freq_bands_names)
            temp_name = sprintf('freq_sum_%s_%s', targets{i_target},freq_bands_names{i_freq});
            temp_name2 = sprintf('freq_mean_%s_%s', targets{i_target},freq_bands_names{i_freq});
            mask_f = freq_bands.(freq_bands_names{i_freq})(1) < f & f <= freq_bands.(freq_bands_names{i_freq})(end);
            for i_subj = 1:numel(features)
                subj_data = features(i_subj).(sub_feature{i_feature});
                temp_data1(i_subj,i_feature,i_freq) = nansum(subj_data(mask_f));
                temp_data2(i_subj,i_feature,i_freq)= nanmean(subj_data(mask_f));
            end
            temp_data1_2=squeeze(mean(temp_data1,2));
            temp_data2_2=squeeze(mean(temp_data2,2));
            for i_subj=1:size(temp_data1_2,1)
                features(i_subj).(temp_name) = temp_data1_2(i_subj,i_freq);
                features(i_subj).(temp_name2) = temp_data2_2(i_subj,i_freq);
            end
        end
        % Calculate ratios between bands
        for i_ratio = 1:numel(ratio_bands_names)
            temp_name3 = sprintf('ratio_sum_%s_%s', targets{i_target},ratio_bands_names{i_ratio});
            temp_name4=sprintf('ratio_mean_%s_%s', targets{i_target},ratio_bands_names{i_ratio});
            mask_low = ratio_bands.(ratio_bands_names{i_ratio})(1,1) < f & f <= ratio_bands.(ratio_bands_names{i_ratio})(1, end);
            mask_high = ratio_bands.(ratio_bands_names{i_ratio})(2,1) < f & f <= ratio_bands.(ratio_bands_names{i_ratio})(2, end);
            for i_subj = 1:numel(features)
                subj_data = features(i_subj).(sub_feature{i_feature});
                temp_data3(i_subj,i_feature,i_ratio) = nansum(subj_data(mask_low)) ./ nansum(subj_data(mask_high));
                temp_data4(i_subj,i_feature,i_ratio) = nanmean(subj_data(mask_low))./ nanmean(subj_data(mask_high));
            end
            temp_data3_2=squeeze(mean(temp_data3,2));
            temp_data4_2=squeeze(mean(temp_data4,2));
            for i_subj=1:size(temp_data3_2,1)
                features(i_subj).(temp_name3) = temp_data3_2(i_subj,i_ratio);
                features(i_subj).(temp_name4) = temp_data4_2(i_subj,i_ratio);
            end
        end
    end
end

%% Get features from cross-correlations
% targets = {'xcorr_'}; % xcorr
% for i_target = 1:numel(targets)
%     str_target = targets{i_target};
%     sub_feature = feature_names(strncmp(feature_names, str_target, numel(str_target)));
%     for i_feature = 1:numel(sub_feature)
%         % Calculate power within bands
%         for i_freq = 1:numel(freq_bands_names)
%             temp_name = sprintf('freq_%s_%s', sub_feature{i_feature}, freq_bands_names{i_freq});
%             mask_f = freq_bands.(freq_bands_names{i_freq})(1) < f & f <= freq_bands.(freq_bands_names{i_freq})(end);
%             for i_subj = 1:numel(features)
%                 subj_data = features(i_subj).(sub_feature{i_feature});
%                 temp_data = nansum(subj_data(mask_f));
%                 features(i_subj).(temp_name) = temp_data;
%             end
%         end
%         % Calculate ratios between bands
%         for i_ratio = 1:numel(ratio_bands_names)
%             temp_name = sprintf('ratio_%s_%s', sub_feature{i_feature}, ratio_bands_names{i_ratio});
%             mask_low = ratio_bands.(ratio_bands_names{i_ratio})(1,1) < f & f <= ratio_bands.(ratio_bands_names{i_ratio})(1, end);
%             mask_high = ratio_bands.(ratio_bands_names{i_ratio})(2,1) < f & f <= ratio_bands.(ratio_bands_names{i_ratio})(2, end);
%             for i_subj = 1:numel(features)
%                 subj_data = features(i_subj).(sub_feature{i_feature});
%                 temp_data = nansum(subj_data(mask_low)) ./ nansum(subj_data(mask_high));
%                 features(i_subj).(temp_name) = temp_data;
%             end
%         end
%     end
% end

