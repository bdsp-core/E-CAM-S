%% Calculation of features that depend on full time
function features = EEG_ICANS_FeaturesFromPSD(features, f)

%% Pull out frequency band info
[freq_bands, ratio_bands] = EEGFreqBandsVariedICANS();
feature_names = fieldnames(features);
freq_bands_names = fieldnames(freq_bands);
ratio_bands_names = fieldnames(ratio_bands);
targets = {'psd_db', 'psd_rel'};
temp_data=zeros(size(features,2),4,size(freq_bands_names,1));
temp_data3=zeros(size(features,2),4,size(ratio_bands_names,1));
for i_target = 1:numel(targets)
    str_target = targets{i_target};
    sub_feature = feature_names(strncmp(feature_names, str_target,numel(str_target)));         %(4 psd names for every lead)
    for i_feature = 1:numel(sub_feature) %4 sub features (leads)
        %Calculate power within bands
        for i_freq = 1:numel(freq_bands_names) %for all frequency bands (26)
            temp_name=sprintf('freq_%s_%s', targets{i_target},freq_bands_names{i_freq});
            mask_f = freq_bands.(freq_bands_names{i_freq})(1) < f & f <= freq_bands.(freq_bands_names{i_freq})(end);
            for i_segs = 1:numel(features)
                segs_data = features(i_segs).(sub_feature{i_feature});              %gives 1x119 matrix (takes power values from feature matrix)
                temp_data(i_segs,i_feature,i_freq) = nansum(segs_data(mask_f));     %gives feature value per lead and for every frequency band
            end
            temp_data2=squeeze(mean(temp_data,2));                                  %take mean channels
            for i_segs=1:size(temp_data2,1)
                features(i_segs).(temp_name) = temp_data2(i_segs,i_freq);
            end
        end
        % Calculate ratios between bands, including "Slow Wave Index" as per http://www.scitepress.org/Papers/2010/29506/29506.pdf
        for i_ratio = 1:numel(ratio_bands_names) %28 band ratio's
            temp_name = sprintf('ratio_%s_%s', targets{i_target},ratio_bands_names{i_ratio});
            mask_low = ratio_bands.(ratio_bands_names{i_ratio})(1,1) < f & f <= ratio_bands.(ratio_bands_names{i_ratio})(1, end);
            mask_high = ratio_bands.(ratio_bands_names{i_ratio})(2,1) < f & f <= ratio_bands.(ratio_bands_names{i_ratio})(2, end);
            for i_segs = 1:numel(features) 
                segs_data = features(i_segs).(sub_feature{i_feature});
                temp_data3(i_segs,i_feature,i_ratio) = nansum(segs_data(mask_low)) ./ nansum(segs_data(mask_high));
            end
            temp_data4=squeeze(mean(temp_data3,2));
             for i_segs=1:size(temp_data4,1)
                features(i_segs).(temp_name) = temp_data4(i_segs,i_ratio);
             end
        end 
    end
end

%% Calculate "Harmonic indexes", e.g. mean spectral frequency, spectral value at mean spectral freq
% http://www.scitepress.org/Papers/2010/29506/29506.pdf
freq_low = 0.5;
freq_high = 30;
mask_f = freq_low <= f & f <= freq_high;
str_target = 'psd_val';
sub_field = feature_names(strncmp(feature_names, str_target, numel(str_target)));
f = f(:)'; % to match orientation of psd
tic;
mean_spec_freq=zeros(size(features,2),4);
meansfpower=zeros(size(features,2),4);
bw=zeros(size(features,2),4);
specent=zeros(size(features,2),4);
sef95=zeros(size(features,2),4);
sef50=zeros(size(features,2),4);
kurtosis_spectral=zeros(size(features,2),4);
for i_subj = 1:numel(features)
    temp_name = sprintf('meanspecfreq_%s',str_target);
    temp_name2 = sprintf('meansfpower_%s',str_target); 
    temp_name3 = sprintf('meansfbw_%s',str_target);  
    temp_name4 = sprintf('specent_%s',str_target);
    temp_name5 = sprintf('sef95_%s',str_target);
    temp_name6 = sprintf('sef50_%s',str_target);
    %temp_name7 = sprintf('kurtosis_spectral%s',str_target);
   % temp_name8 = sprintf('kurtosis_spectral_theta_%s',str_target);
   % temp_name9 = sprintf('kurtosis_spectral_alpha_%s',str_target);
    for i_field = 1:numel(sub_field) %4 subfields for every lead
        psd = features(i_subj).(sub_field{i_field}); %select psd_val for respective EEG lead
        % Mean spectral freq (center frequency)
        mean_spec_freq(i_subj,i_field) = nansum(psd(mask_f) .* f(mask_f)) ./ nansum(psd(mask_f));
        % Power at center freq
        [~,closestIndex] = min(abs(f-mean_spec_freq(i_subj,i_field)));
        meansfpower(i_subj,i_field)=psd(closestIndex);
        % Bandwidth
        bw(i_subj,i_field) = nansum((f - mean_spec_freq(i_subj,i_field) .^ 2 .* psd) / nansum(psd));
        % Spectral Entropy
        temp_psd = psd(mask_f);
        temp_f = f(mask_f);
        prob_psd = temp_psd / sum(temp_psd);
        specent(i_subj,i_field) = pentropy(prob_psd', temp_f, 0);
        % SEF95 + SEF50
        cum_psd = cumsum(prob_psd);
        sef95(i_subj,i_field) = temp_f(find(cum_psd>0.95, 1));
        sef50(i_subj,i_field) = temp_f(find(cum_psd>0.5, 1));
        % Spectral kurtosis
       % kurtosis_spectral(i_subj,i_field)=mean(pkurtosis(psd',200,f,1200));
    end
end
mean_spec_freq2=mean(mean_spec_freq,2);
meansfpower2=mean(meansfpower,2);
bw2=mean(bw,2);
specent2=mean(specent,2);
sef95_2=mean(sef95,2);
sef50_2=mean(sef50,2);
%kurtosis_spectral=mean(kurtosis_spectral,2);
%kurtosis_spectral_theta2=mean(kurtosis_spectral_theta,2);
%kurtosis_spectral_alpha2=mean(kurtosis_spectral_alpha,2);
for i_subj=1:size(mean_spec_freq2,1)
    features(i_subj).(temp_name)=mean_spec_freq2(i_subj,1);
    features(i_subj).(temp_name2) = meansfpower2(i_subj,1);
    features(i_subj).(temp_name3) = bw2(i_subj,1);
    features(i_subj).(temp_name4) = specent2(i_subj,1); 
    features(i_subj).(temp_name5) = sef95_2(i_subj,1);
    features(i_subj).(temp_name6) = sef50_2(i_subj,1);
   % features(i_subj).(temp_name7) = kurtosis_spectral(i_subj,1);
   % features(i_subj).(temp_name8) = kurtosis_spectral_theta2(i_subj,1);
   % features(i_subj).(temp_name9) = kurtosis_spectral_alpha2(i_subj,1);
end     

% https://www-clinicalkey-com.ezp-prod1.hul.harvard.edu/#!/content/playContent/1-s2.0-S0010482512001588?returnurl=null&referrer=null&scrollTo=%23hl0000728


%% Use FOOOF to generate other spectral features: ~1hr for all channels/lead combos
%dirname = [dirDropboxKimchiLab '\MATLAB\fooof-master\fooof_mat-master\fooof_mat'];
%addpath(dirname);

tic;
str_target = 'psd_val_';
sub_field = feature_names(strncmp(feature_names, str_target, numel(str_target)));
for i_subj = 1:numel(features)
    for i_field = 1%:numel(sub_field)
        psd = features(i_subj).(sub_field{i_field});
        fooof_res1 = EEG_CAMS_FOOOF(psd, f);
    end
     for i_field = 2%:numel(sub_field)
        psd = features(i_subj).(sub_field{i_field});
        fooof_res2 = EEG_CAMS_FOOOF(psd, f);
     end
     for i_field = 3%:numel(sub_field)
        psd = features(i_subj).(sub_field{i_field});
        fooof_res3 = EEG_CAMS_FOOOF(psd, f);
     end
     for i_field = 4%:numel(sub_field)
        psd = features(i_subj).(sub_field{i_field});
        fooof_res4 = EEG_CAMS_FOOOF(psd, f);
     end
    fooof_names = fieldnames(fooof_res1);
    fooof_res=[fooof_res1,fooof_res2,fooof_res3,fooof_res4];
    fooof_res=struct2table(fooof_res);
    fooof_res=table2array(fooof_res);
    fooof_res=mean(fooof_res,1); %mean channels
        for i_fooof = 1:numel(fooof_names)
            temp_name = sprintf('fooof_%s', fooof_names{i_fooof});
            features(i_subj).(temp_name) = fooof_res(1,i_fooof);
        end
end
    TimeUpdate(i_subj, numel(features));
end

%% Other features: 
% Multiple segments per pt: 30sec or 1min each?
%
% Time series?
% https://www-sciencedirect-com.ezp-prod1.hul.harvard.edu/science/article/pii/S0169260716308276
% Approximate Entropy?
% Coefficient of Variation: CV?
% Hjorth parameters: The Hjorth parameters provide dynamic
% temporal information of the EEG signal.
% Considering the epoch x, the Hjorth parameters are
% computed from the variance of x, var(x), and the first
% and second derivatives x’, x’’ according to (AnsariAsl et al, 2007)
% Activity = var(x) (7)
% Mobility = var(x') var(x) (8)
% 2 Complexity = var(x'')× var(x) var(x') . (9)
% Entropy:
% The entropy gives a measure of signal disorder
% and can provide relevant information in the detection
% of some sleep disturbs. It is computed from
% histogram of the EEG samples of each sleep epoch,
% according with (Zoubek et al, 2007)
% Kurtosis, Skew, STD, Var, RMS, Per75/etc: https://www-sciencedirect-com.ezp-prod1.hul.harvard.edu/science/article/pii/S0165027015000230
% 
% Post time series?
% Log transform data/ratios? https://www.ncbi.nlm.nih.gov/pubmed/12089718


