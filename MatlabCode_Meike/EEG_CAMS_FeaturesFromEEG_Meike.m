%% Calculation of features that depend on full time
function features = EEG_CAMS_FeaturesFromEEG_Meike(bipolar_data, bipolar_abbrev, Fs, filename, tbl_clin)
  %bipolar_data is segmental data
%% Check that data is valid
% if isempty(bipolar_data)
%     features = []; % Make class to make sure have blank data where needed?
%     return;
% end

%% Time Domain Features: Some of these could be band specific rather than just lead specific?
% Mean
% Var
% Std
% Skewness
% Kurtosis
% Threshold Percentile
% Median

%bipolar_data=squeeze(segs_wa(1,:,:));
time.mean = mean(mean(bipolar_data, 2));
time.meandiff = mean(mean(abs(diff(bipolar_data, 1, 2)), 2)); % e.g. mean "line-length"
time.var = mean(var(bipolar_data, 0, 2)); % also = Hjorth's "Activity"
time.std = mean(std(bipolar_data, 0, 2));
time.skew = mean(skewness(bipolar_data, 0, 2));
time.kurtosis_time = mean(kurtosis(bipolar_data, 0, 2));
time.median = mean(median(bipolar_data, 2));
time.per25 = mean(prctile(bipolar_data, 25, 2));
time.per75 = mean(prctile(bipolar_data, 75, 2));
% Zero Crossing Rate (ZCR): 
temp_data = bsxfun(@minus, bipolar_data, mean(bipolar_data, 2));
temp_data = diff(temp_data > 0, 1, 2);
time.zerocross = mean(sum(temp_data ~= 0, 2));

% Hjorth parameters: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/59158/versions/1/previews/calculate_features.m/index.html
% Mobility
diff_eeg = diff(bipolar_data, 1, 2);
std_diff_eeg = std(diff_eeg, 0, 2);
time.hjorthmobility = mean(std_diff_eeg ./ time.std);
% Complexity
time.hjorthcomplexity = mean(std(diff(diff_eeg, 1, 2), 0, 2)./ std_diff_eeg ./ time.hjorthmobility);

% mean absolute gradient from Haoqi's paper: --> have to check
abs2=abs(gradient(bipolar_data));
time.gradient=mean(nanmean(abs2,2));

%% Shannon Entropy & Fractal dimension: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/59158/versions/1/previews/calculate_features.m/index.html
num_ch=size(bipolar_data,1);
time.sentropy = mean(nan(num_ch, 1));
time.fractaldim = mean(nan(num_ch, 3));

for i_ch = 1:num_ch
    time.sentropy(i_ch, :) = wentropy(bipolar_data(i_ch, :), 'shannon');
    time.fractaldim(i_ch, :) = wfbmesti(bipolar_data(i_ch, :));
end
time.sentropy=mean(time.sentropy);
time.fractaldim=mean(time.fractaldim);
time.fractaldim=mean(time.fractaldim,2);
% https://www.mathworks.com/matlabcentral/fileexchange/50289-a-set-of-entropy-measures-for-temporal-series-1d-signals

%% Make features: Time domain
%features.dur_bins = ts; % This will be a column vector, ok since not using for analysis
time_names = fieldnames(time);
for i_time = 1:numel(time_names)
    %for i_ch = 1:numel(bipolar_abbrev)
        temp_field = sprintf('time_%s_%s', time_names{i_time});%,bipolar_abbrev{i_ch});
        features.(temp_field) = time.(time_names{i_time});%(i_ch, :);
   % end
end

%% Calculate spectrograms / psd for filtered data
num_sec = 6;
win_samples = Fs*num_sec;
frac_overlap = 1 - (1/num_sec);
num_ch=size(bipolar_data,1);
clear p;
for i_ch = 1:num_ch
   temp_data = bipolar_data(i_ch, :);
    [~, f, t, p(:, :, i_ch)] = spectrogram(temp_data, win_samples, floor(win_samples*frac_overlap),[0.5:0.25:30],Fs);
     %[~, f, t, p(:, :, i_ch)] = spectrogram(temp_data, [], [],[0.5:0.25:30],Fs);
end
psd.val = squeeze(mean(p, 2));  % Given state changes during recording, mean is probably better (awake to drowsy)
psd.db = SpecDb(psd.val);       % Given state changes during recording, mean is probably better (awake to drowsy)
%psd.std = squeeze(std(p, 0, 2));
%psd.iqr = squeeze(iqr(p, 2));

% crop_f = [1 30]; % From Slooter at al papers, e.g. BJA 2018
% mask_f = crop_f(1) <= f & f <= crop_f(end);
crop_freq = [0.5 30];
crop_freq2= [0.5 20];
mask_f = crop_freq(1) <= f & f <= crop_freq(end);
mask_f2 = crop_freq2(1) <= f & f <= crop_freq2(end);
psd.rel = bsxfun(@rdivide, psd.val, sum(psd.val(mask_f, :))); % relative psd
%psd.rel2= bsxfun(@rdivide, psd.val, sum(psd.val(mask_f2, :))); % relative psd based on BAI index

%% Make features: Spectral (on)
% %crop_freq = [0.1 40]; % Haoqi sleep paper used 0.5-20 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6251659/
features.f = f(mask_f); % This will be a column vector, ok since not using for analysis

% For regression learning, best to have everything as a row vector, rather than column
psd_names = fieldnames(psd);
for i_ch = 1:numel(bipolar_abbrev)
    for i_psd = 1:numel(psd_names)
        temp_field = sprintf('psd_%s_%s', psd_names{i_psd}, bipolar_abbrev{i_ch});
        features.(temp_field) = psd.(psd_names{i_psd})(mask_f, i_ch)';
    end
end

%% Generate some coherence values for features (on)
% tic
%unipolar_names=unipolar_names(1,[1, 2, 4, 5]);
%unipolar_data=unipolar_data([1,2,4,5],:);
for i_lead = 1:numel(bipolar_abbrev)
    x = bipolar_data(i_lead, :);
    for j_lead = i_lead:numel(bipolar_abbrev)
        y = bipolar_data(j_lead, :);
        [cxy, f] = mscohere(x, y, [], [], [0.5:0.25:30], Fs);
        temp_field = sprintf('cohere_%s_%s', bipolar_abbrev{i_lead}, bipolar_abbrev{j_lead});
        %mask_f2 = crop_freq(1) <= f & f <= crop_freq(end);
        features.(temp_field) = cxy';

        % Cross-Correlation
%         temp_field = sprintf('xcorr_%s_%s', unipolar_names{i_lead}, unipolar_names{j_lead});
%         [xc, lags] = xcorr(x, y, Fs, 'coeff'); %normalized ccf (so ccfn in [-1,1])
%         features.(temp_field) = xc; %cross-correlation
%         features.xc_lags = lags;
    end
%     TimeUpdate(i_lead, numel(unipolar_names));

%% Haoqi's features
% 95th percentile, 5th percentile, min, mean, standard deviation of relative delta band power	3 × 4	Delta power/total power between 0.5 Hz and 20 Hz
% 95th percentile, 5th percentile, min, mean, standard deviation of relative theta band power	3 × 4	Theta power/total power between 0.5 Hz and 20 Hz
% 95th percentile, 5th percentile, min, mean, standard deviation of relative alpha band power	3 × 4	Alpha power/total power between 0.5 Hz and 20 Hz
% 95th percentile, 5th percentile, min, mean, standard deviation of delta-theta power ratio	3 × 4	Delta power/theta power
% 95th percentile, 5th percentile, min, mean, standard deviation of delta-alpha power ratio	3 × 4	Delta power/alpha power
% 95th percentile, 5th percentile, min, mean, standard deviation of theta-alpha power ratio	3 × 4	Theta power/alpha power
freq_bands.delta = [0.5 4];
freq_bands.theta = [4 8];
freq_bands.alpha = [8 12];
freq_bands.beta = [13 20];
freq_bands.deltaVtheta = [freq_bands.delta; freq_bands.theta];
freq_bands.deltaValpha = [freq_bands.delta; freq_bands.alpha];
freq_bands.thetaValpha = [freq_bands.theta; freq_bands.alpha];
name_bands = fieldnames(freq_bands);
num_bands = numel(name_bands);
band_all = [0.5 20];
mask_f_all = band_all(1) <= f & f < band_all(end);
data_all = squeeze(sum(p(mask_f_all, :, :)));
for i_band = 1:num_bands
    temp_band = freq_bands.(name_bands{i_band});
    mask_f_num = temp_band(1, 1) <= f & f < temp_band(1, 2);
    data_band_num = squeeze(sum(p(mask_f_num, :, :)));
    if numel(temp_band) == 2
        data_rel = data_band_num ./ data_all;
    else
        mask_f_denom = temp_band(2, 1) <= f & f < temp_band(2, 2);
        data_band_denom = squeeze(sum(p(mask_f_denom, :, :)));
        data_rel = data_band_num ./ data_band_denom;
    end
    spec.min = mean(min(data_rel));
    spec.max = mean(max(data_rel));
    spec.per5 = mean(prctile(data_rel, 5));
    spec.per95 = mean(prctile(data_rel, 95));
    spec.median = mean(median(data_rel));
    spec.iqr = mean(iqr(data_rel));
    spec.mean = mean(mean(data_rel));
    spec.std = mean(std(data_rel));
    spec_names = fieldnames(spec);
    for i_ch = 1:numel(bipolar_abbrev)
        for i_spec = 1:numel(spec_names)
            temp_field = sprintf('spec_%s_%s', name_bands{i_band}, spec_names{i_spec});
            features.(temp_field) = spec.(spec_names{i_spec});
        end
    end
end
end

%% Generate some correlation values: Between unipolar chan and ref?
% In case so globally coherent average ref eliminated all important coherence?

% For a select subset of features
% features.lead_cohere = {
%     'F3_C3', 'F4_C4' % Front back parasaggital: Ant
%     'C3_P3', 'C4_P4' % Front back parasaggital: Post
%     'F3_P3', 'F4_P4' % Front back parasaggital
%     'F3_P4', 'F4_P3' % Diagonal back parasaggital
%     'F3_T5', 'F4_T6' % Outward parasaggital
%     'F3_Fz', 'F4_Fz' % Pyramid/Triangle: front to back
%     'F3_Cz', 'F4_Cz' % Pyramid/Triangle: front to back
%     'F3_Pz', 'F4_Pz' % Pyramid/Triangle: front to back
%     'Fz_P3', 'Fz_P4' % Pyramid/Triangle: front to back
%     'Cz_P3', 'Cz_P4' % Pyramid/Triangle: front to back
%     'P3_Pz', 'P4_Pz' % Pyramid/Triangle: front to back
%     'F3_F4', 'P3_P4' % Front contra vs. back contra
%     'Fz_Cz', 'Cz_Pz' % Midline
%     'Fp1_O1', 'Fp2_O2' % Front back parasaggital: Farther away
%     'F7_O1', 'F8_O2' % Front back parasaggital: Farther away
%     'F3_O1', 'F4_O2' % Front back parasaggital: Farther away
%     'F7_Pz', 'F8_Pz'
%     'Fp1_F7', 'Fp2_F8' % Forehead
%     'T3_T5', 'T4_T6'
%     'F7_T5', 'F8_T6' % Most lateral
%     'Pz_T3', 'Pz_T4'
%     'Fz_T5', 'Fz_T6'
% };
% 
% tic
% for i_pair = 1:size(features.lead_cohere, 1)
%     mask_x = strcmp(chan_names, features.lead_cohere{i_pair, 1});
%     x = eeg_data(mask_x, :);
%     mask_y = strcmp(chan_names, features.lead_cohere{i_pair, end});
%     y = eeg_data(mask_y, :);
% %     if sum(mask_x) == 0 || sum(mask_y) == 0
% %         fprintf('leads not found\n');
% %         features.lead_cohere{i_pair, :}
% %     else
%     [cxy, f] = mscohere(x, y, win_samples, [], [], Fs);
%     temp_field = sprintf('cohere_%s_VS_%s', features.lead_cohere{i_pair, 1}, features.lead_cohere{i_pair, end});
%     features.(temp_field) = cxy(mask_f)';
%     
%     % Cross-Correlation
%     temp_field = sprintf('xcorr_%s_VS_%s', features.lead_cohere{i_pair, 1}, features.lead_cohere{i_pair, end});
%     [xc, lags] = xcorr(x, y, Fs);
%     features.(temp_field) = xc;
%     features.xc_lags = lags;
% %     end
%     TimeUpdate(i_pair, size(features.lead_cohere, 1));
% end

